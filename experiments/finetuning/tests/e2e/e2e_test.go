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

	h := NewHarness(t, "finetuning-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/finetuning")

	// Build images
	h.DockerBuild("finetuning-server:e2e", filepath.Join(experimentRoot, "images/server/Dockerfile"), experimentRoot)
	h.DockerBuild("finetuning-client:e2e", filepath.Join(experimentRoot, "images/client/Dockerfile"), experimentRoot)

	// Load images into Kind
	h.KindLoad("finetuning-server:e2e")
	h.KindLoad("finetuning-client:e2e")

	// Read manifest and replace placeholders
	manifestPath := filepath.Join(experimentRoot, "k8s/manifest.yaml")
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("Failed to read manifest: %v", err)
	}
	manifest := string(b)
	manifest = strings.ReplaceAll(manifest, "SERVER_IMAGE_PLACEHOLDER", "finetuning-server:e2e")
	manifest = strings.ReplaceAll(manifest, "CLIENT_IMAGE_PLACEHOLDER", "finetuning-client:e2e")
	manifest = strings.ReplaceAll(manifest, "imagePullPolicy: IfNotPresent", "imagePullPolicy: Never")
	// Add imagePullPolicy: Never if not present
	manifest = strings.ReplaceAll(manifest, "image: finetuning-server:e2e", "image: finetuning-server:e2e\n        imagePullPolicy: Never")
	manifest = strings.ReplaceAll(manifest, "image: finetuning-client:e2e", "image: finetuning-client:e2e\n        imagePullPolicy: Never")

	// Reduce resource requirements for E2E
	manifest = strings.ReplaceAll(manifest, "cpu: \"2\"", "cpu: \"500m\"")
	manifest = strings.ReplaceAll(manifest, "memory: \"8Gi\"", "memory: \"2Gi\"")
	manifest = strings.ReplaceAll(manifest, "cpu: \"4\"", "cpu: \"1\"")
	manifest = strings.ReplaceAll(manifest, "memory: \"16Gi\"", "memory: \"4Gi\"")

	// Apply manifest
	h.DeleteJob("finetuning-client")
	h.DeleteDeployment("finetuning-server")
	h.DeleteService("finetuning-server")
	h.KubectlApplyContent(manifest)

	// Wait for server
	h.WaitForDeployment("finetuning-server", 5*time.Minute)

	// Wait for client job
	err = h.WaitForJobSuccess("finetuning-client", 10*time.Minute)

	// Check logs (always, even on failure)
	logs := h.GetPodLogs("app=finetuning-server")
	t.Logf("Server logs:\n%s", logs)

	clientLogs := h.GetPodLogs("batch.kubernetes.io/job-name=finetuning-client")
	t.Logf("Client logs:\n%s", clientLogs)

	if err != nil {
		t.Fatalf("Job failed: %v", err)
	}

	if !strings.Contains(clientLogs, "Fine-tuning completed successfully") {
		t.Error("Client logs do not indicate successful completion")
	}
}
