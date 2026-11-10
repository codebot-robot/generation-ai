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

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func TestCmdPlanEmit(t *testing.T) {
	tmpDir := t.TempDir()

	manifest := `{
		"source": {
			"commit_sha": "commit_test_123"
		},
		"tensors": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
		},
		"config": {
			"model_type": "deepseek_v41",
			"n_layers": 1,
			"dim": 128
		}
	}`
	binding := `{
		"bound": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
		}
	}`

	artDir := filepath.Join(tmpDir, "artifact")
	if err := os.MkdirAll(artDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artDir, "manifest.json"), []byte(manifest), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artDir, "binding.json"), []byte(binding), 0644); err != nil {
		t.Fatal(err)
	}

	planPath := filepath.Join(tmpDir, "plan.json")
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", planPath, "--rows", "1,5,8"}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("ReadFile plan failed: %v", err)
	}

	var plan kernelplan.Plan
	if err := json.Unmarshal(planBytes, &plan); err != nil {
		t.Fatalf("Unmarshal plan failed: %v", err)
	}
	if plan.Format != kernelplan.PlanFormat {
		t.Errorf("plan format = %q, want %q", plan.Format, kernelplan.PlanFormat)
	}
	if plan.TreeCommit != "commit_test_123" {
		t.Errorf("plan tree_commit = %q, want commit_test_123", plan.TreeCommit)
	}
	if len(plan.Sites) == 0 {
		t.Error("expected at least one site in plan")
	}

	if _, err := kernelplan.HashPlan(&plan); err != nil {
		t.Fatalf("HashPlan failed: %v", err)
	}
}
