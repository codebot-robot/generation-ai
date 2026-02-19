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

package e2e

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestE2E(t *testing.T) {
	if os.Getenv("RUN_E2E") == "" {
		t.Skip("Skipping E2E test; RUN_E2E not set")
	}

	h := NewHarness(t, "inference-test-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/inference-test")

	// Build image
	h.DockerBuild("inference-test:e2e", filepath.Join(experimentRoot, "images/inference-test/Dockerfile"), experimentRoot)

	// Load image into Kind
	h.KindLoad("inference-test:e2e")

	// 1. Run Single-node CPU Test
	t.Run("SingleNodeCPU", func(t *testing.T) {
		h.DeleteJob("inference-test-single")
		manifestPath := filepath.Join(experimentRoot, "k8s/manifest.yaml")
		b, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("Failed to read manifest: %v", err)
		}
		manifest := string(b)
		manifest = strings.ReplaceAll(manifest, "name: inference-test", "name: inference-test-single")
		manifest = strings.ReplaceAll(manifest, "image: IMAGE_PLACEHOLDER", "image: inference-test:e2e\n        imagePullPolicy: Never")

		// Remove GPU requirement for CPU test
		manifest = strings.ReplaceAll(manifest, "nvidia.com/gpu: 1", "cpu: \"500m\"")
		manifest = strings.ReplaceAll(manifest, "cloud.google.com/gke-accelerator: nvidia-l4", "")
		// Add small memory limit
		manifest = strings.ReplaceAll(manifest, "resources:", "resources:\n          requests:\n            memory: \"2Gi\"\n          limits:\n            memory: \"4Gi\"")

		h.KubectlApplyContent("inference-test-single", manifest)
		err = h.WaitForJobSuccess("inference-test-single", 10*time.Minute)

		logs := h.GetPodLogs("batch.kubernetes.io/job-name=inference-test-single")
		t.Logf("Single-node logs:\n%s", logs)

		if err != nil {
			t.Fatalf("Job failed: %v", err)
		}
		if !strings.Contains(logs, "Tokens generated:") {
			t.Error("Logs do not contain expected metrics")
		}
	})

	// 2. Run Distributed CPU Test (FSDP requested but fallback)
	t.Run("DistributedCPU", func(t *testing.T) {
		h.DeleteJob("inference-test")
		h.DeleteService("inference-test-headless")
		manifestPath := filepath.Join(experimentRoot, "k8s/manifest-distributed.yaml")
		b, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Log("manifest-distributed.yaml not found, skipping distributed test")
			return
		}
		manifest := string(b)
		manifest = strings.ReplaceAll(manifest, "image: IMAGE_PLACEHOLDER", "image: inference-test:e2e\n        imagePullPolicy: Never")

		// Tweak for CPU
		manifest = strings.ReplaceAll(manifest, "nvidia.com/gpu: 1", "cpu: \"500m\"")
		manifest = strings.ReplaceAll(manifest, "cloud.google.com/gke-accelerator: nvidia-l4", "")
		// Use smaller model for E2E
		manifest = strings.ReplaceAll(manifest, "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")

		h.KubectlApplyContent("inference-test-distributed", manifest)
		err = h.WaitForJobSuccess("inference-test", 10*time.Minute)

		logs := h.GetPodLogs("app=inference-test")
		t.Logf("Distributed logs:\n%s", logs)

		if err != nil {
			t.Fatalf("Job failed: %v", err)
		}
		if !strings.Contains(logs, "Tokens generated:") {
			t.Error("Logs do not contain expected metrics")
		}
	})
}
