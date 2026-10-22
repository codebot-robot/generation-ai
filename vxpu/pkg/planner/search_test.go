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
	"math"
	"testing"
)

// A layout's score is the candidate's prediction, for the shapes both
// know: the general form prices by the same rules.
func TestScore_AgreesWithPredict(t *testing.T) {
	for _, spec := range [][]string{
		{"g2-standard-24", "c3d-standard-60", "c3d-standard-60"},
		{"g2-standard-48", "g2-standard-48"},
	} {
		fleet := fleetOf(t, spec...)
		s := NewScorer(deepseekFull(), fleet, nil)
		for _, c := range Candidates(deepseekFull(), fleet, nil) {
			if c.Infeasible != "" {
				continue
			}
			sc := s.Score(s.LayoutOf(c))
			if !sc.Feasible || math.Abs(sc.Ms-c.Prediction.MsPerToken) > 0.02*c.Prediction.MsPerToken {
				t.Errorf("%v %s: score %.1f (%v, %s), candidate %.1f", spec, c.Summary, sc.Ms, sc.Feasible, sc.Why, c.Prediction.MsPerToken)
			}
		}
	}
}

// skewedCounts is traffic where a quarter of each layer's experts serve
// most selections, as recorded on DeepSeek (#317).
func skewedCounts() map[int]map[int]int64 {
	counts := map[int]map[int]int64{}
	for layer := range 40 {
		counts[layer] = map[int]int64{}
		for e := range 384 {
			n := int64(1)
			if (e*7+layer)%4 == 0 {
				n = 12
			}
			counts[layer][e] = n
		}
	}
	return counts
}

// With recorded traffic too: GPU experts on the machines that run their
// layers, placed and priced by the same rules.
func TestScore_AgreesWithPredictWithGPUExperts(t *testing.T) {
	counts := skewedCounts()
	for _, spec := range [][]string{
		{"g2-standard-24", "c3d-standard-60", "c3d-standard-60"},
		{"g2-standard-48", "g2-standard-48"},
		{"g4-standard-96"},
	} {
		fleet := fleetOf(t, spec...)
		s := NewScorer(deepseekFull(), fleet, counts)
		for _, c := range Candidates(deepseekFull(), fleet, counts) {
			if c.Infeasible != "" {
				continue
			}
			sc := s.Score(s.LayoutOf(c))
			t.Logf("%v %s: score %.1f, candidate %.1f", spec, c.Summary, sc.Ms, c.Prediction.MsPerToken)
			if !sc.Feasible || math.Abs(sc.Ms-c.Prediction.MsPerToken) > 0.02*c.Prediction.MsPerToken {
				t.Errorf("%v %s: score %.1f (%v, %s), candidate %.1f", spec, c.Summary, sc.Ms, sc.Feasible, sc.Why, c.Prediction.MsPerToken)
			}
		}
	}
}

// The search finds, on its own, the layouts the candidates rank first.
func TestSearch_Finds(t *testing.T) {
	for _, c := range []struct {
		spec   []string
		counts map[int]map[int]int64
		kind   string
	}{
		{[]string{"g2-standard-24", "c3d-standard-60", "c3d-standard-60"}, nil, "experts-spread"},
		{[]string{"g2-standard-48", "g2-standard-48", "c3d-standard-60", "c3d-standard-60"}, nil, "layer-ranges (2)"},
		{[]string{"g2-standard-48", "g2-standard-48", "c3d-standard-60", "c3d-standard-60"}, skewedCounts(), "layer-ranges (2)"},
	} {
		fleet := fleetOf(t, c.spec...)
		s := NewScorer(deepseekFull(), fleet, c.counts)
		sr := NewSearch(s, 10, 1)
		for range 20000 {
			sr.Step()
		}
		best := sr.Top[0]
		want := Candidates(deepseekFull(), fleet, c.counts)[0]
		t.Logf("%v: best %s %.1f ms (%s) after %d layouts, %d feasible; candidates' best %.1f", c.spec, best.Kind, best.Ms,
			s.Describe(best.Layout), sr.Scanned, sr.Valid, want.Prediction.MsPerToken)
		if best.Kind != c.kind || best.Ms > want.Prediction.MsPerToken*1.02 {
			t.Errorf("%v: the search found %s at %.1f ms, want %s at about %.1f", c.spec, best.Kind, best.Ms, c.kind, want.Prediction.MsPerToken)
		}
	}
}

// TestSearch_FewLayers: a model with fewer layers than the most ranges a
// random layout makes (four) panicked choosing its cuts.
func TestSearch_FewLayers(t *testing.T) {
	a := deepseekFull()
	a.Experts = map[int]Layer{0: a.Experts[0], 1: a.Experts[1]}
	a.Engram = nil
	fleet := fleetOf(t, "g2-standard-48", "g2-standard-48", "c3d-standard-60", "c3d-standard-60")
	sr := NewSearch(NewScorer(a, fleet, nil), 10, 1)
	for range 500 {
		sr.Step()
	}
}
