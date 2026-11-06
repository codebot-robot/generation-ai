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
	"math"
	"os"
	"testing"
)

func TestFixtureSideBranchNotCountingAndFusedCountsOnce(t *testing.T) {
	data, err := os.ReadFile("testdata/trace_side_branch_fused.json")
	if err != nil {
		t.Fatalf("read fixture trace: %v", err)
	}

	var trace struct {
		Format  string      `json:"format"`
		Shape   string      `json:"shape"`
		Rows    int         `json:"rows"`
		Streams int         `json:"streams"`
		Nodes   []PhaseNode `json:"nodes"`
	}
	if err := json.Unmarshal(data, &trace); err != nil {
		t.Fatalf("unmarshal fixture trace: %v", err)
	}

	model := NewCriticalPathCostModel()
	pred, err := model.Predict(trace.Nodes, Shape{Rows: trace.Rows, Streams: trace.Streams})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	// 1. Verify side branch node does not count towards the wall.
	var sideNode *CriticalPathNodeSummary
	for i := range pred.CriticalNodes {
		if pred.CriticalNodes[i].Name == "model.layers.0.compressor_island" {
			sideNode = &pred.CriticalNodes[i]
			break
		}
	}
	if sideNode == nil {
		t.Fatalf("side branch node model.layers.0.compressor_island missing from critical nodes")
	}
	if sideNode.WallCostMs != 0.0 {
		t.Errorf("expected side branch node wall cost 0.0 ms, got %.2f ms", sideNode.WallCostMs)
	}
	if math.Abs(pred.SideBranchCost-0.0) > 1e-6 {
		t.Errorf("expected total side branch wall cost 0.0 ms, got %.2f ms", pred.SideBranchCost)
	}
	if math.Abs(pred.HiddenWorkMs-0.7) > 1e-6 {
		t.Errorf("expected hidden work 0.7 ms, got %.2f ms", pred.HiddenWorkMs)
	}

	// 2. Verify fused region counts once in place and constituent parts are skipped.
	var parentCount, partSkippedCount int
	for _, n := range pred.CriticalNodes {
		if n.FusedRegion == "fused_push_0" {
			if n.WallCostMs > 0 {
				parentCount++
				if math.Abs(n.WallCostMs-0.5) > 1e-6 {
					t.Errorf("expected in-place fused cost 0.5 ms, got %.2f ms", n.WallCostMs)
				}
			}
			if n.FusedPartSkipped {
				partSkippedCount++
			}
		}
	}
	if parentCount != 1 {
		t.Errorf("expected fused region to count exactly once, got %d times", parentCount)
	}
	if partSkippedCount != 3 {
		t.Errorf("expected 3 fused parts skipped, got %d", partSkippedCount)
	}
	if math.Abs(pred.FusedSavingsMs-0.20) > 1e-6 {
		t.Errorf("expected fused savings 0.20 ms, got %.2f ms", pred.FusedSavingsMs)
	}

	// 3. Verify total predicted step time:
	// 2.0 (dense_input) + 2.0 (attn_q) + 5.0 (moe_experts) + 2.0 (collective_sum) + 0.5 (fused_push) = 11.5 ms.
	expectedTotal := 11.5
	if math.Abs(pred.PredictedStepMs-expectedTotal) > 1e-6 {
		t.Errorf("expected predicted step time %.2f ms, got %.2f ms", expectedTotal, pred.PredictedStepMs)
	}

	// Collective wait check: wait_ms was 1.2 ms on collective_sum.
	if math.Abs(pred.CollectiveWait-1.2) > 1e-6 {
		t.Errorf("expected collective wait 1.2 ms, got %.2f ms", pred.CollectiveWait)
	}
}

func TestSideBranchTouchingWindowWallCost(t *testing.T) {
	// Main A: [0, 2] ms
	// Side S: [1, 5] ms (duration 4 ms)
	// Main B: [5, 7] ms (dependent consumer starts when S ends at 5 ms)
	// Overlap: Main A covers [1, 2] ms (1 ms). Main B touches at 5 ms (0 ms covered).
	// Uncovered wall cost of Side S: 4 - 1 = 3 ms.
	// Total wall clock: 2 (A) + 2 (B) + 3 (S) = 7 ms.
	nodes := []PhaseNode{
		{
			Name:       "main_a",
			SideBranch: false,
			StartNs:    0,
			EndNs:      2_000_000,
			DurationMs: 2.0,
		},
		{
			Name:       "side_s",
			SideBranch: true,
			StartNs:    1_000_000,
			EndNs:      5_000_000,
			DurationMs: 4.0,
		},
		{
			Name:       "main_b",
			SideBranch: false,
			StartNs:    5_000_000,
			EndNs:      7_000_000,
			DurationMs: 2.0,
		},
	}

	model := NewCriticalPathCostModel()
	pred, err := model.Predict(nodes, Shape{Rows: 1, Streams: 1})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	if math.Abs(pred.SideBranchCost-3.0) > 1e-6 {
		t.Errorf("expected side branch cost 3.0 ms, got %.2f ms", pred.SideBranchCost)
	}
	if math.Abs(pred.HiddenWorkMs-1.0) > 1e-6 {
		t.Errorf("expected hidden work 1.0 ms, got %.2f ms", pred.HiddenWorkMs)
	}
	if math.Abs(pred.PredictedStepMs-7.0) > 1e-6 {
		t.Errorf("expected predicted step time 7.0 ms, got %.2f ms", pred.PredictedStepMs)
	}
}

func TestStrictRowMismatchRefused(t *testing.T) {
	nodes := []PhaseNode{
		{Name: "dense_node", Kind: "dense", DurationMs: 1.0, Rows: 1},
	}
	model := NewCriticalPathCostModel()
	_, err := model.Predict(nodes, Shape{Rows: 5, Streams: 1})
	if err == nil {
		t.Fatalf("expected error when predicting 5-row shape using 1-row node, got nil")
	}
}

func TestFusedRegionValidation(t *testing.T) {
	model := NewCriticalPathCostModel()

	t.Run("missing parent refused", func(t *testing.T) {
		nodes := []PhaseNode{
			{Name: "part1", DurationMs: 0.3, FusedRegion: "reg1", IsFusedParent: false},
		}
		if _, err := model.Predict(nodes, Shape{Rows: 1, Streams: 1}); err == nil {
			t.Fatalf("expected error for fused region with no parent, got nil")
		}
	})

	t.Run("multiple parents refused", func(t *testing.T) {
		nodes := []PhaseNode{
			{Name: "parent1", DurationMs: 0.5, FusedRegion: "reg1", IsFusedParent: true},
			{Name: "parent2", DurationMs: 0.5, FusedRegion: "reg1", IsFusedParent: true},
		}
		if _, err := model.Predict(nodes, Shape{Rows: 1, Streams: 1}); err == nil {
			t.Fatalf("expected error for fused region with multiple parents, got nil")
		}
	})

	t.Run("parent zero duration refused", func(t *testing.T) {
		nodes := []PhaseNode{
			{Name: "parent", DurationMs: 0.0, FusedRegion: "reg1", IsFusedParent: true},
		}
		if _, err := model.Predict(nodes, Shape{Rows: 1, Streams: 1}); err == nil {
			t.Fatalf("expected error for fused region with zero duration parent, got nil")
		}
	})
}

func TestEmptyNodesRefused(t *testing.T) {
	model := NewCriticalPathCostModel()
	if _, err := model.Predict(nil, Shape{Rows: 1, Streams: 1}); err == nil {
		t.Fatalf("expected error for empty nodes list, got nil")
	}
}
