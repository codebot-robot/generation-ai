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
	"fmt"
	"sort"
)

// PhaseNode represents one stamped execution node in an in-graph phase trace.
type PhaseNode struct {
	Name          string  `json:"name"`
	Class         string  `json:"class,omitempty"`
	Kind          string  `json:"kind,omitempty"`
	SideBranch    bool    `json:"side_branch,omitempty"`
	StartNs       int64   `json:"start_ns,omitempty"`
	EndNs         int64   `json:"end_ns,omitempty"`
	DurationMs    float64 `json:"duration_ms"`
	Rows          int     `json:"rows,omitempty"`
	FusedRegion   string  `json:"fused_region,omitempty"`
	IsFusedParent bool    `json:"is_fused_parent,omitempty"`
	WaitMs        float64 `json:"wait_ms,omitempty"`
}

// EffectiveClass returns the node's classification category.
func (n *PhaseNode) EffectiveClass() string {
	if n.Class != "" {
		return n.Class
	}
	if n.Kind != "" {
		return n.Kind
	}
	return "dense"
}

// Shape describes the serving configuration: row count and stream count.
type Shape struct {
	Rows    int `json:"rows"`
	Streams int `json:"streams"`
}

// CriticalPathNodeSummary records how a single node was evaluated.
type CriticalPathNodeSummary struct {
	Name             string  `json:"name"`
	Class            string  `json:"class"`
	SideBranch       bool    `json:"side_branch"`
	RawCostMs        float64 `json:"raw_cost_ms"`
	WallCostMs       float64 `json:"wall_cost_ms"`
	OverlappedMs     float64 `json:"overlapped_ms,omitempty"`
	FusedRegion      string  `json:"fused_region,omitempty"`
	FusedPartSkipped bool    `json:"fused_part_skipped,omitempty"`
	CollectiveWaitMs float64 `json:"collective_wait_ms,omitempty"`
}

// CriticalPathPrediction holds the cost model's predictions.
type CriticalPathPrediction struct {
	Rows            int                       `json:"rows"`
	Streams         int                       `json:"streams"`
	PredictedStepMs float64                   `json:"predicted_step_ms"`
	CriticalNodes   []CriticalPathNodeSummary `json:"critical_nodes,omitempty"`
	ByClassMs       map[string]float64        `json:"by_class_ms"`
	SideBranchCost  float64                   `json:"side_branch_cost_ms"`
	HiddenWorkMs    float64                   `json:"hidden_work_ms"`
	FusedSavingsMs  float64                   `json:"fused_savings_ms"`
	CollectiveWait  float64                   `json:"collective_wait_ms"`
}

// CriticalPathCostModel predicts step time per plan and shape from in-graph stamps.
type CriticalPathCostModel struct {
	StrictRowCheck bool
}

// NewCriticalPathCostModel creates a new critical path cost model v0.
func NewCriticalPathCostModel() *CriticalPathCostModel {
	return &CriticalPathCostModel{StrictRowCheck: true}
}

// Predict calculates the predicted step time on the critical path.
func (m *CriticalPathCostModel) Predict(nodes []PhaseNode, shape Shape) (*CriticalPathPrediction, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("cost model: cannot predict step time from empty nodes list")
	}

	if m.StrictRowCheck && shape.Rows > 0 {
		for _, n := range nodes {
			if n.Rows > 0 && n.Rows != shape.Rows {
				return nil, fmt.Errorf("cost model: node %q recorded at %d rows, but plan shape runs %d rows: knobs change sign across row count",
					n.Name, n.Rows, shape.Rows)
			}
		}
	}

	fusedRegions, err := validateFusedRegions(nodes)
	if err != nil {
		return nil, err
	}

	pred := &CriticalPathPrediction{
		Rows:      shape.Rows,
		Streams:   shape.Streams,
		ByClassMs: make(map[string]float64),
	}
	for _, info := range fusedRegions {
		if info.partsSum > info.parentCost {
			pred.FusedSavingsMs += (info.partsSum - info.parentCost)
		}
	}

	var mainSpans []timeSpan
	for _, n := range nodes {
		if !n.SideBranch && n.EndNs > n.StartNs {
			mainSpans = append(mainSpans, timeSpan{start: n.StartNs, end: n.EndNs})
		}
	}

	for _, n := range nodes {
		class := n.EffectiveClass()
		if n.FusedRegion != "" && !n.IsFusedParent {
			pred.CriticalNodes = append(pred.CriticalNodes, CriticalPathNodeSummary{
				Name:             n.Name,
				Class:            class,
				SideBranch:       n.SideBranch,
				RawCostMs:        n.DurationMs,
				WallCostMs:       0,
				FusedRegion:      n.FusedRegion,
				FusedPartSkipped: true,
			})
			continue
		}

		wallCost := n.DurationMs
		var overlappedMs float64
		if n.SideBranch {
			coveredNs := coveredDurationNs(n.StartNs, n.EndNs, mainSpans)
			uncoveredNs := (n.EndNs - n.StartNs) - coveredNs
			if uncoveredNs < 0 {
				uncoveredNs = 0
			}
			wallCost = float64(uncoveredNs) / 1e6
			overlappedMs = n.DurationMs - wallCost
			if overlappedMs < 0 {
				overlappedMs = 0
			}
			pred.SideBranchCost += wallCost
			pred.HiddenWorkMs += overlappedMs
		}

		if n.WaitMs > 0 {
			pred.CollectiveWait += n.WaitMs
		}

		pred.PredictedStepMs += wallCost
		pred.ByClassMs[class] += wallCost
		pred.CriticalNodes = append(pred.CriticalNodes, CriticalPathNodeSummary{
			Name:             n.Name,
			Class:            class,
			SideBranch:       n.SideBranch,
			RawCostMs:        n.DurationMs,
			WallCostMs:       wallCost,
			OverlappedMs:     overlappedMs,
			FusedRegion:      n.FusedRegion,
			CollectiveWaitMs: n.WaitMs,
		})
	}

	return pred, nil
}

type fusedRegionInfo struct {
	parentCount int
	parentCost  float64
	partsSum    float64
}

func validateFusedRegions(nodes []PhaseNode) (map[string]*fusedRegionInfo, error) {
	regions := make(map[string]*fusedRegionInfo)
	for _, n := range nodes {
		if n.FusedRegion == "" {
			continue
		}
		info := regions[n.FusedRegion]
		if info == nil {
			info = &fusedRegionInfo{}
			regions[n.FusedRegion] = info
		}
		if n.IsFusedParent {
			info.parentCount++
			info.parentCost = n.DurationMs
		} else {
			info.partsSum += n.DurationMs
		}
	}
	for reg, info := range regions {
		if info.parentCount == 0 {
			return nil, fmt.Errorf("cost model: fused region %q has no parent node", reg)
		}
		if info.parentCount > 1 {
			return nil, fmt.Errorf("cost model: fused region %q has %d parent nodes; expected exactly one", reg, info.parentCount)
		}
		if info.parentCost <= 0 {
			return nil, fmt.Errorf("cost model: fused region %q parent has non-positive duration %.2f ms", reg, info.parentCost)
		}
	}
	return regions, nil
}

type timeSpan struct {
	start int64
	end   int64
}

func coveredDurationNs(sideStart, sideEnd int64, mainSpans []timeSpan) int64 {
	if sideEnd <= sideStart {
		return 0
	}
	var clipped []timeSpan
	for _, m := range mainSpans {
		cStart := m.start
		if cStart < sideStart {
			cStart = sideStart
		}
		cEnd := m.end
		if cEnd > sideEnd {
			cEnd = sideEnd
		}
		if cStart < cEnd {
			clipped = append(clipped, timeSpan{start: cStart, end: cEnd})
		}
	}
	if len(clipped) == 0 {
		return 0
	}
	sort.Slice(clipped, func(i, j int) bool {
		return clipped[i].start < clipped[j].start
	})
	var merged []timeSpan
	cur := clipped[0]
	for _, s := range clipped[1:] {
		if s.start <= cur.end {
			if s.end > cur.end {
				cur.end = s.end
			}
		} else {
			merged = append(merged, cur)
			cur = s
		}
	}
	merged = append(merged, cur)
	var total int64
	for _, s := range merged {
		total += s.end - s.start
	}
	return total
}
