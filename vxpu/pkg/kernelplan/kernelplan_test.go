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
	"slices"
	"testing"
)

func TestEmitFromArtifact(t *testing.T) {
	manifest := `{
		"tensors": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.layers.0.mlp.down_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.embed_tokens.weight": {"length": 1000, "shape": [1000, 128], "dtype": "F8_E4M3"},
			"model.layers.0.input_layernorm.weight": {"length": 128, "shape": [128], "dtype": "F8_E4M3"},
			"e0.w1": {"length": 100}, "e1.w1": {"length": 100}
		},
		"config": {
			"model_type": "deepseek_v41",
			"n_layers": 1,
			"dim": 128,
			"n_heads": 4,
			"head_dim": 32,
			"n_routed_experts": 8,
			"moe_inter_dim": 64,
			"engram_layer_ids": [0],
			"engram_vocab_size": 1000
		}
	}`
	binding := `{
		"bound": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.layers.0.mlp.down_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.embed_tokens.weight": {"length": 1000, "shape": [1000, 128], "dtype": "F8_E4M3"},
			"model.layers.0.input_layernorm.weight": {"length": 128, "shape": [128], "dtype": "F8_E4M3"}
		},
		"external": {
			"experts": {
				"0": {
					"w1": {"parts": [["e0.w1"], ["e1.w1"]]}
				}
			},
			"engram": {
				"0": {}
			}
		}
	}`

	plan, err := EmitFromArtifact([]byte(manifest), []byte(binding), "commit123", []int{1, 5, 8})
	if err != nil {
		t.Fatalf("EmitFromArtifact failed: %v", err)
	}

	if plan.Format != PlanFormat {
		t.Errorf("plan.Format = %q, want %q", plan.Format, PlanFormat)
	}
	if plan.TreeCommit != "commit123" {
		t.Errorf("plan.TreeCommit = %q, want commit123", plan.TreeCommit)
	}
	wantMD5 := ComputeArtifactMD5([]byte(manifest), []byte(binding))
	if plan.ArtifactMD5 != wantMD5 {
		t.Errorf("plan.ArtifactMD5 = %q, want %q", plan.ArtifactMD5, wantMD5)
	}

	classesSeen := make(map[string]int)
	for _, s := range plan.Sites {
		classesSeen[s.Class]++
		if !slices.Equal(s.RowCounts, []int{1, 5, 8}) {
			t.Errorf("site %s row counts = %v, want [1, 5, 8]", s.ID, s.RowCounts)
		}
		if s.Strategy != "default" {
			t.Errorf("site %s strategy = %q, want default", s.ID, s.Strategy)
		}
		if s.Class == ClassDenseGEMV {
			if s.ID == "model.embed_tokens.weight" || s.ID == "model.layers.0.input_layernorm.weight" {
				t.Errorf("non-linear weight %s should not be emitted as dense GEMV", s.ID)
			}
		}
	}

	for _, c := range []string{ClassDenseGEMV, ClassExpertOp, ClassSumSite, ClassAttention, ClassDrafter} {
		if classesSeen[c] == 0 {
			t.Errorf("expected at least one site of class %q, got 0", c)
		}
	}
}

func TestEmitTreeCommitFromManifest(t *testing.T) {
	manifest := `{
		"source": {
			"commit_sha": "manifest_commit_sha_123"
		},
		"config": {"dim": 128}
	}`
	binding := `{}`

	plan, err := EmitFromArtifact([]byte(manifest), []byte(binding), "", []int{1, 5})
	if err != nil {
		t.Fatalf("EmitFromArtifact failed: %v", err)
	}
	if plan.TreeCommit != "manifest_commit_sha_123" {
		t.Errorf("plan.TreeCommit = %q, want manifest_commit_sha_123", plan.TreeCommit)
	}

	// Unknown / empty tree commit without manifest commit fails
	badManifest := `{"config": {"dim": 128}}`
	if _, err := EmitFromArtifact([]byte(badManifest), []byte(binding), "", []int{1, 5}); err == nil {
		t.Error("expected error when tree commit cannot be determined, got nil")
	}
}
