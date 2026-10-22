// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package planner chooses the physical plan the coordinator sends with a
// load (#394): from what the artifact holds, the machine the executor
// runs on and what the model's traffic selected, the switches the
// executor's plan reads, each choice with its reason.
//
// v0 chooses one thing: which routed experts the GPUs hold. It is
// expert_stats.placement moved into the coordinator, with the budgets
// derived from the machine instead of given by a person.
package planner

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

const (
	// gpuHeadroomBytes is kept free on every GPU: activations, the KV
	// cache, CUDA's own allocations.
	gpuHeadroomBytes = 8 << 30
	// denseFactor is what the engine's GPU holds besides its experts, as
	// a multiple of the artifact's dense weight bytes: the weights, the
	// bf16 copies of the fp8 ones (the engine's bf16 weight cache), the KV
	// cache and the CUDA graph pools. Measured on DeepSeek-V4.1-Flash on
	// two RTX PRO 6000s (the hero run, 68 + 95 GB of experts): GPU0 used
	// 24 GiB besides its experts, for 9.8 GB of dense weights.
	denseFactor = 2.5
	// ramFraction and ramHeadroomBytes give the RAM a machine's experts
	// may take: the rest is the process, the kernels' buffers, the page
	// cache the n-gram tables are read through (they are memory-mapped
	// and page in as used, #364, so they are not counted here).
	ramFraction      = 0.85
	ramHeadroomBytes = 16 << 30
)

// Artifact is what a compiled model holds, as the planner needs it.
type Artifact struct {
	DenseBytes int64         // weights bound into the graphs
	Experts    map[int]Layer // routed experts held outside the graphs, by layer
}

// Layer is one layer's routed experts.
type Layer struct {
	Experts        int
	BytesPerExpert int64
}

// ReadArtifact takes the inventory of an artifact's manifest and binding.
func ReadArtifact(manifestJSON, bindingJSON []byte) (Artifact, error) {
	var manifest struct {
		Tensors map[string]struct {
			Length int64 `json:"length"`
		} `json:"tensors"`
	}
	var binding struct {
		Bound map[string]struct {
			Length int64 `json:"length"`
		} `json:"bound"`
		Stacked map[string]struct {
			Parts [][]string `json:"parts"`
		} `json:"stacked"`
		External struct {
			Experts map[string]map[string]struct {
				Parts [][]string `json:"parts"`
			} `json:"experts"`
		} `json:"external"`
	}
	if err := json.Unmarshal(manifestJSON, &manifest); err != nil {
		return Artifact{}, fmt.Errorf("manifest: %w", err)
	}
	if err := json.Unmarshal(bindingJSON, &binding); err != nil {
		return Artifact{}, fmt.Errorf("binding: %w", err)
	}
	bytesOf := func(parts [][]string) (int64, error) {
		var n int64
		for _, part := range parts {
			for _, name := range part {
				t, ok := manifest.Tensors[name]
				if !ok {
					return 0, fmt.Errorf("binding names tensor %q, which the manifest lacks", name)
				}
				n += t.Length
			}
		}
		return n, nil
	}
	a := Artifact{Experts: map[int]Layer{}}
	for _, t := range binding.Bound {
		a.DenseBytes += t.Length
	}
	for name, s := range binding.Stacked {
		n, err := bytesOf(s.Parts)
		if err != nil {
			return Artifact{}, fmt.Errorf("stacked %s: %w", name, err)
		}
		a.DenseBytes += n
	}
	for key, banks := range binding.External.Experts {
		layer, err := strconv.Atoi(key)
		if err != nil {
			return Artifact{}, fmt.Errorf("experts of layer %q: %w", key, err)
		}
		var l Layer
		for bank, spec := range banks {
			if l.Experts != 0 && len(spec.Parts) != l.Experts {
				return Artifact{}, fmt.Errorf("layer %d: bank %s has %d experts, another %d", layer, bank, len(spec.Parts), l.Experts)
			}
			if len(spec.Parts) == 0 {
				return Artifact{}, fmt.Errorf("layer %d: bank %s has no experts", layer, bank)
			}
			l.Experts = len(spec.Parts)
			n, err := bytesOf(spec.Parts)
			if err != nil {
				return Artifact{}, fmt.Errorf("layer %d bank %s: %w", layer, bank, err)
			}
			l.BytesPerExpert += n / int64(len(spec.Parts))
		}
		a.Experts[layer] = l
	}
	return a, nil
}

// Plan is what the planner chose: the switches to send (LoadModelRequest
// plan), the layers other executors serve, and why.
type Plan struct {
	Switches map[string]string `json:"switches"`
	// ExpertExecutors is the layers each other executor (by address)
	// serves to the engine's mailboxes (ServeExperts).
	ExpertExecutors map[string][]int `json:"expert_executors,omitempty"`
	Reasons         []string         `json:"reasons"`
}

// Executor is another executor the coordinator may give experts to.
type Executor struct {
	Address string
	Probe   machines.Probe
}

// Choose plans a load of artifact for an engine on the machine probe
// describes (nil: unknown), with others the free executors it may spread
// the experts over, given the model's recorded decode selections (layer
// -> expert -> count). The experts go to RAM first, the engine's and then
// the others' (spread), then the hottest of the engine's to its GPUs.
func Choose(a Artifact, probe *machines.Probe, others []Executor, counts map[int]map[int]int64) Plan {
	p := Plan{Switches: map[string]string{}}
	if len(a.Experts) == 0 {
		p.Reasons = append(p.Reasons, "experts: the model has no routed experts held outside its graphs")
		return p
	}
	local := spread(&p, a, probe, others)
	for layer := range counts {
		if !local[layer] {
			delete(counts, layer) // served elsewhere: the engine's GPUs cannot hold them
		}
	}
	placeOnGPUs(&p, a, probe, counts)
	return p
}

// spread decides which layers' experts the engine holds in its RAM and
// which go to other executors' RAM: the engine keeps the layers that fit
// its budget, in order; the next layers fill each other executor's; any
// beyond every budget stay on the engine (paged from its disk). Returns
// the layers the engine holds.
func spread(p *Plan, a Artifact, probe *machines.Probe, others []Executor) map[int]bool {
	layers := make([]int, 0, len(a.Experts))
	var total int64
	for layer, l := range a.Experts {
		layers = append(layers, layer)
		total += int64(l.Experts) * l.BytesPerExpert
	}
	sort.Ints(layers)
	local := map[int]bool{}
	for _, layer := range layers {
		local[layer] = true
	}
	if probe == nil || probe.RAMBytes == 0 {
		p.Reasons = append(p.Reasons, "experts: all on the engine (it did not say how much RAM it has)")
		return local
	}
	budget := ramBudget(probe.RAMBytes)
	if probe.GPU == "" {
		budget -= a.DenseBytes // the dense weights are in RAM too
	}
	next := 0 // the first layer not yet held in RAM
	take := func(budget int64) (held []int, bytes int64) {
		for ; next < len(layers); next++ {
			l := a.Experts[layers[next]]
			size := int64(l.Experts) * l.BytesPerExpert
			if bytes+size > budget {
				break
			}
			held, bytes = append(held, layers[next]), bytes+size
		}
		return held, bytes
	}
	kept, keptBytes := take(budget)
	if next == len(layers) {
		p.Reasons = append(p.Reasons, fmt.Sprintf("experts: all %d layers (%.1f GB) in the engine's RAM (budget %.1f GB of its %.1f GB)",
			len(layers), gb(total), gb(budget), gb(probe.RAMBytes)))
		return local
	}
	p.Reasons = append(p.Reasons, fmt.Sprintf("experts: %d of %d layers (%.1f of %.1f GB) in the engine's RAM (budget %.1f GB of its %.1f GB)",
		len(kept), len(layers), gb(keptBytes), gb(total), gb(budget), gb(probe.RAMBytes)))
	for _, other := range others {
		if next == len(layers) {
			break
		}
		held, bytes := take(ramBudget(other.Probe.RAMBytes))
		if len(held) == 0 {
			continue
		}
		if p.ExpertExecutors == nil {
			p.ExpertExecutors = map[string][]int{}
		}
		p.ExpertExecutors[other.Address] = held
		for _, layer := range held {
			delete(local, layer)
		}
		p.Reasons = append(p.Reasons, fmt.Sprintf("experts: layers %d-%d (%.1f GB) on %s (%s, %.1f GB RAM), answering the engine's mailboxes",
			held[0], held[len(held)-1], gb(bytes), other.Address, describe(other.Probe), gb(other.Probe.RAMBytes)))
	}
	if next < len(layers) {
		var short int64
		for _, layer := range layers[next:] {
			short += int64(a.Experts[layer].Experts) * a.Experts[layer].BytesPerExpert
		}
		p.Reasons = append(p.Reasons, fmt.Sprintf("experts: %d layers (%.1f GB) fit no machine's RAM: they stay on the engine, paged from its disk",
			len(layers)-next, gb(short)))
	}
	if len(p.ExpertExecutors) > 0 {
		p.Switches["VXPU_EXPERT_MAILBOXES"] = "1" // the other executors answer through them
	}
	return local
}

func ramBudget(ram int64) int64 { return int64(ramFraction*float64(ram)) - ramHeadroomBytes }

func describe(p machines.Probe) string {
	if m := machines.Match(p); m != nil {
		return m.Name
	}
	if p.GPU != "" {
		return fmt.Sprintf("%dx %s", p.GPUCount, p.GPU)
	}
	return p.CPU
}

// placeOnGPUs chooses which of the engine's experts its GPUs hold: the
// hottest by counts (which hold only the engine's layers).
func placeOnGPUs(p *Plan, a Artifact, probe *machines.Probe, counts map[int]map[int]int64) {
	switch {
	case probe == nil:
		p.Reasons = append(p.Reasons, "GPU experts: none (the executor did not describe its machine)")
		return
	case probe.GPU == "" || probe.GPUCount == 0:
		p.Reasons = append(p.Reasons, "GPU experts: none (the executor has no GPU)")
		return
	case len(counts) == 0:
		p.Reasons = append(p.Reasons, "GPU experts: none (no recorded selections for the engine's layers yet: "+
			"a placement needs its traffic; requests record it)")
		return
	}
	var perExpert int64 // the largest layer's: every placed expert must fit
	for _, l := range a.Experts {
		perExpert = max(perExpert, l.BytesPerExpert)
	}
	budgets := make([]int64, probe.GPUCount)
	for i := range budgets {
		budgets[i] = probe.GPUMemoryBytes - gpuHeadroomBytes
	}
	budgets[0] -= int64(denseFactor * float64(a.DenseBytes))
	for i := range budgets { // a GPU with less room than one expert (or none: the headroom) holds none
		if budgets[i] < perExpert {
			budgets[i] = 0
		}
	}
	placement, held, share := place(counts, perExpert, budgets)
	if held == 0 {
		p.Reasons = append(p.Reasons, fmt.Sprintf("GPU experts: none (no GPU has room for a %.2f GB expert)", gb(perExpert)))
		return
	}
	data, err := json.Marshal(placement)
	if err != nil {
		p.Reasons = append(p.Reasons, fmt.Sprintf("GPU experts: none (the placement did not encode: %v)", err))
		return
	}
	p.Switches["VXPU_GPU_EXPERTS"] = string(data)
	total := 0
	for _, l := range a.Experts {
		total += l.Experts
	}
	budgetText := ""
	for i, b := range budgets {
		budgetText += fmt.Sprintf(" GPU%d %.1f GB", i, gb(b))
	}
	p.Reasons = append(p.Reasons, fmt.Sprintf("GPU experts: %d of %d on %d %s GPU(s), serving %.0f%% of the recorded selections; "+
		"budgets%s (each GPU's %.1f GB, less %.1f GB headroom; GPU0 also less %.1fx the %.1f GB of dense weights)",
		held, total, probe.GPUCount, probe.GPU, 100*share, budgetText, gb(probe.GPUMemoryBytes), gb(gpuHeadroomBytes),
		denseFactor, gb(a.DenseBytes)))
}

// place is expert_stats.placement: the experts selected most, across
// layers, fill the budgets in order, the hottest on the engine's GPU
// (every expert costs the same, so the most selections per byte). The
// result is a vxpu-placement/2 document; held is how many experts it
// holds, share the fraction of the selections they served.
func place(counts map[int]map[int]int64, perExpert int64, budgets []int64) (map[string]any, int, float64) {
	type ranked struct {
		n             int64
		layer, expert int
	}
	var all []ranked
	var total int64
	for layer, c := range counts {
		for expert, n := range c {
			all = append(all, ranked{n, layer, expert})
			total += n
		}
	}
	sort.Slice(all, func(i, j int) bool { // as Python sorts (n, layer, expert) descending
		if all[i].n != all[j].n {
			return all[i].n > all[j].n
		}
		if all[i].layer != all[j].layer {
			return all[i].layer > all[j].layer
		}
		return all[i].expert > all[j].expert
	})
	var gpus []map[string]any
	start := 0
	var served int64
	for _, budget := range budgets {
		end := start
		if perExpert > 0 && budget > 0 {
			end = min(len(all), start+int(budget/perExpert))
		}
		layers := map[string][]int{}
		for _, r := range all[start:end] {
			key := strconv.Itoa(r.layer)
			layers[key] = append(layers[key], r.expert)
			served += r.n
		}
		for _, ids := range layers {
			sort.Ints(ids)
		}
		gpus = append(gpus, map[string]any{"experts": end - start, "layers": layers})
		start = end
	}
	history := map[string]map[string]int64{} // the engine's scores when experts move (experts_gpu.Mover)
	for layer, c := range counts {
		history[strconv.Itoa(layer)] = map[string]int64{}
		for expert, n := range c {
			history[strconv.Itoa(layer)][strconv.Itoa(expert)] = n
		}
	}
	share := 0.0
	if total > 0 {
		share = float64(served) / float64(total)
	}
	return map[string]any{"format": "vxpu-placement/2", "gpus": gpus, "counts": history}, start, share
}

func gb(n int64) float64 { return float64(n) / 1e9 }
