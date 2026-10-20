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

package planner

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

func TestReadArtifact(t *testing.T) {
	manifest := `{"tensors": {
		"e0.w1": {"length": 100}, "e1.w1": {"length": 100},
		"e0.w2": {"length": 50},  "e1.w2": {"length": 50},
		"s0": {"length": 7}, "s1": {"length": 7}}}`
	binding := `{"bound": {"embed": {"length": 1000}},
		"stacked": {"bank": {"parts": [["s0"], ["s1"]]}},
		"external": {"experts": {"3": {
			"w1": {"parts": [["e0.w1"], ["e1.w1"]]},
			"w2": {"parts": [["e0.w2"], ["e1.w2"]]}}}}}`
	a, err := ReadArtifact([]byte(manifest), []byte(binding))
	if err != nil {
		t.Fatal(err)
	}
	if a.DenseBytes != 1014 || a.Experts[3] != (Layer{Experts: 2, BytesPerExpert: 150}) {
		t.Errorf("got %+v", a)
	}
	if _, err := ReadArtifact([]byte(`{"tensors": {}}`), []byte(binding)); err == nil {
		t.Error("a binding naming tensors the manifest lacks should be refused")
	}
	empty := `{"external": {"experts": {"3": {"w1": {"parts": []}}}}}`
	if _, err := ReadArtifact([]byte(manifest), []byte(empty)); err == nil {
		t.Error("a bank with no experts should be refused")
	}
}

// deepseek is DeepSeek-V4.1-Flash's inventory: 40 layers of 384 experts
// of 18.8 MB, 9.8 GB of dense weights.
func deepseek() Artifact {
	a := Artifact{DenseBytes: 9_800_000_000, Experts: map[int]Layer{}}
	for layer := range 40 {
		a.Experts[layer] = Layer{Experts: 384, BytesPerExpert: 18_800_000}
	}
	return a
}

// uniform is traffic that selected every expert of every layer, the
// lower ids more.
func uniform(a Artifact) map[int]map[int]int64 {
	counts := map[int]map[int]int64{}
	for layer, l := range a.Experts {
		counts[layer] = map[int]int64{}
		for e := range l.Experts {
			counts[layer][e] = int64(1000 - e)
		}
	}
	return counts
}

func TestChoose_DeepSeekOnTwoRTXPro6000(t *testing.T) {
	a := deepseek()
	g4 := &machines.Probe{Device: "cuda", GPU: "NVIDIA RTX PRO 6000", GPUMemoryBytes: 102_005_473_280, GPUCount: 2}
	p := Choose(a, g4, uniform(a))
	var placement struct {
		Gpus []struct {
			Experts int
			Layers  map[string][]int
		}
	}
	if err := json.Unmarshal([]byte(p.Switches["VXPU_GPU_EXPERTS"]), &placement); err != nil {
		t.Fatalf("placement: %v (plan %+v)", err, p)
	}
	// the budgets the hero run used by hand: about 68 GB on GPU0 (it also
	// holds the dense weights), 95 GB on GPU1
	var held []int
	for _, g := range placement.Gpus {
		held = append(held, g.Experts)
	}
	if len(held) != 2 || held[0] != 3665 || held[1] != 4968 {
		t.Errorf("placed %v experts per GPU, want 3665 (68.9 GB) and 4968 (93.4 GB)", held)
	}
	// the hottest on the engine's GPU: expert 0 of every layer
	if ids := placement.Gpus[0].Layers["7"]; len(ids) == 0 || ids[0] != 0 {
		t.Errorf("GPU0 should hold the hottest experts, layer 7 holds %v", ids)
	}
	if len(p.Reasons) != 1 || !strings.Contains(p.Reasons[0], "8633 of 15360") {
		t.Errorf("reasons: %v", p.Reasons)
	}
}

func TestChoose_ReasonsForTheCPU(t *testing.T) {
	a := deepseek()
	gpu := &machines.Probe{Device: "cuda", GPU: "NVIDIA L4", GPUMemoryBytes: 24e9, GPUCount: 1}
	for _, c := range []struct {
		name     string
		artifact Artifact
		probe    *machines.Probe
		counts   map[int]map[int]int64
		reason   string
	}{
		{"dense model", Artifact{DenseBytes: 1e9}, gpu, uniform(a), "no routed experts"},
		{"undescribed executor", a, nil, uniform(a), "did not describe"},
		{"CPU executor", a, &machines.Probe{Device: "cpu", CPU: "AMD EPYC 9B14"}, uniform(a), "no GPU"},
		{"no traffic yet", a, gpu, nil, "no recorded selections"},
		{"dense weights fill the GPU", Artifact{DenseBytes: 20e9, Experts: a.Experts}, gpu, uniform(a), "no GPU has room"},
	} {
		p := Choose(c.artifact, c.probe, c.counts)
		if len(p.Switches) != 0 || len(p.Reasons) != 1 || !strings.Contains(p.Reasons[0], c.reason) {
			t.Errorf("%s: got %+v, want no switches and a reason about %q", c.name, p, c.reason)
		}
	}
}

// A GPU after the first with less memory than the headroom: its budget
// went negative and place sliced backwards, a panic in the router's
// request handler.
func TestChoose_SmallSecondGPU(t *testing.T) {
	a := deepseek()
	probe := machines.Probe{CPU: "test", CPUThreads: 8, RAMBytes: 64 << 30, GPU: "small", GPUCount: 2, GPUMemoryBytes: 4 << 30}
	if p := Choose(a, &probe, uniform(a)); p.Switches["VXPU_GPU_EXPERTS"] != "" {
		t.Errorf("experts placed on GPUs with no room: %v", p.Reasons)
	}
}

// Experts of no size (a bank whose parts read as empty) divided by zero.
func TestPlace_NoExpertSize(t *testing.T) {
	place(map[int]map[int]int64{0: {0: 1}}, 0, []int64{1 << 30, -1})
}
