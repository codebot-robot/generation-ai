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

package kernelplan

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

// Options tune the policy.
type Options struct {
	// LaunchUs is one kernel launch inside a CUDA graph, credited to a
	// body the compiler fuses into its neighbours (its first operation
	// fuses into what produces its input). 0: no credit. The engine's own
	// costing measures it per card (costs.py: about 2 us); a record's
	// stats may carry it as "launch_us".
	LaunchUs float64
}

// group is every record site that shares an op and a shape key: the unit
// the measurements are pooled over.
type group struct {
	op, shapes string
	sites      []Site
	times      map[string][]float64 // body -> measured microseconds, over the records
	fused      map[string]bool
}

// Plan reads records of one GPU and writes the table for it. Records of
// different GPUs are refused: a table is per card.
func Plan(ctx context.Context, records []*Record, opts Options) (*Table, error) {
	log := klog.FromContext(ctx)
	if len(records) == 0 {
		return nil, fmt.Errorf("no records to plan from")
	}
	gpu := records[0].Hardware.GPU
	for _, r := range records[1:] {
		if r.Hardware.GPU != gpu {
			return nil, fmt.Errorf("records of different GPUs (%q and %q): a table is per card", gpu, r.Hardware.GPU)
		}
	}
	t := &Table{Format: TableFormat, GPU: gpu, Knobs: map[string]any{}}
	launch := opts.LaunchUs
	if launch == 0 {
		launch = statNumber(records, "launch_us")
	}

	groups := map[string]*group{}
	var order []string
	for _, r := range records {
		for _, s := range r.Sites {
			if s.Source == "kept" || s.Chosen == "" {
				continue // not lowered: nothing to choose
			}
			k := s.Op + "\x00" + s.Shapes
			g := groups[k]
			if g == nil {
				g = &group{op: s.Op, shapes: s.Shapes, times: map[string][]float64{}, fused: map[string]bool{}}
				groups[k] = g
				order = append(order, k)
			}
			g.sites = append(g.sites, s)
			for body, us := range s.MeasuredUs {
				g.times[body] = append(g.times[body], us)
			}
			for _, f := range s.Fused {
				g.fused[f] = true
			}
		}
	}
	sort.Strings(order)
	for _, k := range order {
		g := groups[k]
		body, why := g.choose(launch)
		if perSite := g.disagreeing(body); len(perSite) > 0 {
			// the measurement is per shape, but the engine chose differently at
			// some sites (a table given before, say): say so per site rather
			// than hide it under one entry
			for _, s := range g.sites {
				t.Sites = append(t.Sites, Entry{Op: g.op, Site: s.Site, Impl: body, Why: why})
			}
			t.Explain = append(t.Explain, fmt.Sprintf("%s %s: %s; the record ran %s at %s",
				shortOp(g.op), g.shapes, why, strings.Join(perSite, ", "), "some sites, so the table names every site"))
			continue
		}
		t.Sites = append(t.Sites, Entry{Op: g.op, Shapes: g.shapes, Impl: body, Why: why})
		t.Explain = append(t.Explain, fmt.Sprintf("%s %s (%s): %s", shortOp(g.op), g.shapes, plural(len(g.sites), "site"), why))
	}

	// the knobs: native formats from the card, the rest from sweeps, else as the record ran
	if major := capabilityMajor(records[0].Hardware.GPUCapability); major > 0 {
		native := major >= 10 // Blackwell and later take MXFP4 on the tensor cores
		t.Knobs["FP4_NATIVE"] = native
		t.Explain = append(t.Explain, fmt.Sprintf("FP4_NATIVE=%v: compute capability %s", native, records[0].Hardware.GPUCapability))

		warps := 4
		if major >= 10 {
			warps = 8
		}
		t.Knobs["FRONT_SPLIT_WARPS"] = warps
		t.Explain = append(t.Explain, fmt.Sprintf("FRONT_SPLIT_WARPS=%d: compute capability %s", warps, records[0].Hardware.GPUCapability))
	}
	sweeps := map[string]map[string]float64{}
	for _, r := range records {
		if raw, ok := r.Stats["sweeps"]; ok {
			var s map[string]map[string]float64
			if err := json.Unmarshal(raw, &s); err != nil {
				return nil, fmt.Errorf("record stats.sweeps: %w", err)
			}
			for knob, values := range s {
				if sweeps[knob] == nil {
					sweeps[knob] = map[string]float64{}
				}
				for v, us := range values {
					sweeps[knob][v] = us
				}
			}
		}
	}
	for knob, values := range sweeps {
		if _, known := knobTypes[knob]; !known {
			log.Info("ignoring a sweep of a knob the engine does not know", "knob", knob)
			continue
		}
		best, bestUs := "", math.Inf(1)
		for v, us := range values {
			if us < bestUs || (us == bestUs && v < best) {
				best, bestUs = v, us
			}
		}
		value, err := knobValue(knob, best)
		if err != nil {
			return nil, fmt.Errorf("record stats.sweeps: %w", err)
		}
		t.Knobs[knob] = value
		t.Explain = append(t.Explain, fmt.Sprintf("%s=%s: the fastest of %d swept values (%.1f us)", knob, best, len(values), bestUs))
	}
	for knob, value := range records[len(records)-1].Knobs {
		if _, set := t.Knobs[knob]; set || value == nil {
			continue
		}
		if _, known := knobTypes[knob]; !known {
			continue
		}
		t.Knobs[knob] = value // as the last record ran: nothing measured says otherwise
	}
	log.Info("planned the kernel table", "gpu", gpu, "sites", len(t.Sites), "knobs", len(t.Knobs))
	return t, nil
}

// choose is the policy for one group: the measured-fastest body, a launch
// credited to a fused one; without measurements, what the record ran (the
// engine's registry order) when the sites agree.
func (g *group) choose(launch float64) (string, string) {
	if len(g.times) > 0 {
		best, bestCost := "", math.Inf(1)
		parts := make([]string, 0, len(g.times))
		for body, samples := range g.times {
			cost := mean(samples)
			if g.fused[body] {
				cost -= launch
			}
			parts = append(parts, fmt.Sprintf("%s %.1f us", body, mean(samples)))
			if cost < bestCost || (cost == bestCost && body < best) {
				best, bestCost = body, cost
			}
		}
		sort.Strings(parts)
		why := "measured: " + strings.Join(parts, ", ")
		if launch > 0 {
			why += fmt.Sprintf(" (a fused body credited %.1f us for the launch it saves)", launch)
		}
		return best, why
	}
	counts := map[string]int{}
	for _, s := range g.sites {
		counts[s.Chosen]++
	}
	best := ""
	for body, n := range counts {
		if best == "" || n > counts[best] || (n == counts[best] && body < best) {
			best = body
		}
	}
	return best, "not measured: as the engine ran it (its registry's order)"
}

// disagreeing lists the sites whose recorded choice is not body.
func (g *group) disagreeing(body string) []string {
	var out []string
	for _, s := range g.sites {
		if s.Chosen != body {
			out = append(out, s.Site+" ran "+s.Chosen)
		}
	}
	return out
}

// Mismatch is a record site that did not run the table's body.
type Mismatch struct {
	Site Site
	Want string
}

func (m Mismatch) String() string {
	where := m.Site.Site
	if where == "" {
		where = m.Site.Shapes
	}
	return fmt.Sprintf("%s at %s ran %s, the table names %s", shortOp(m.Site.Op), where, m.Site.Chosen, m.Want)
}

// Check compares what a record ran against a table: every site the table
// names must have run the named body. A planner whose choices are not
// applied is no planner, and the failure is silent otherwise (#490).
func Check(t *Table, r *Record) []Mismatch {
	var out []Mismatch
	for _, s := range r.Sites {
		e := t.Entry(s.Op, s.Site, s.Shapes)
		if e != nil && e.Impl != s.Chosen {
			out = append(out, Mismatch{Site: s, Want: e.Impl})
		}
	}
	return out
}

func statNumber(records []*Record, name string) float64 {
	for _, r := range records {
		if raw, ok := r.Stats[name]; ok {
			var f float64
			if json.Unmarshal(raw, &f) == nil {
				return f
			}
		}
	}
	return 0
}

// knobValue parses a swept value's text as the knob's type.
func knobValue(knob, text string) (any, error) {
	switch knobTypes[knob] {
	case "bool":
		switch text {
		case "true", "1":
			return true, nil
		case "false", "0":
			return false, nil
		}
		return nil, fmt.Errorf("knob %s: %q is not true or false", knob, text)
	case "int":
		n, err := strconv.Atoi(text)
		if err != nil {
			return nil, fmt.Errorf("knob %s: %w", knob, err)
		}
		if knob == "FRONT_SPLIT_WARPS" && !isSupportedWarps(n) {
			return nil, fmt.Errorf("knob FRONT_SPLIT_WARPS: unsupported warp count %d", n)
		}
		return n, nil
	}
	return nil, fmt.Errorf("unknown knob %q", knob)
}

func mean(xs []float64) float64 {
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

func plural(n int, noun string) string {
	if n == 1 {
		return "1 " + noun
	}
	return fmt.Sprintf("%d %ss", n, noun)
}

func shortOp(op string) string {
	_, name, found := strings.Cut(op, "::")
	if !found {
		return op
	}
	return name
}
