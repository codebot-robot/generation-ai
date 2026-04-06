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
	"fmt"
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

	h := NewHarness(t, "bema-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/bema")

	// Build images
	h.DockerBuild("bema:e2e", filepath.Join(experimentRoot, "images/bema/Dockerfile"), experimentRoot)

	// Load images into Kind
	h.KindLoad("bema:e2e")

	// Read manifest and replace placeholders
	manifestPath := filepath.Join(experimentRoot, "k8s/manifest.yaml")
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("Failed to read manifest: %v", err)
	}
	manifest := string(b)
	manifest = strings.ReplaceAll(manifest, "image: bema:latest", "image: bema:e2e\n          imagePullPolicy: Never")

	// Apply manifests
	h.DeleteService("bema", "bema")
	h.DeleteStatefulSet("bema", "bema")

	h.KubectlApplyContent("bema", manifest)

	// Wait for server
	if err := h.WaitForStatefulSet("bema", "bema", 2*time.Minute); err != nil {
		fmt.Fprintf(os.Stderr, "Bema failed to start: %v\n", err)
		fmt.Fprintf(os.Stderr, "Bema Pod YAML:\n%s\n", h.GetPodYaml("app=bema", "bema"))
		fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("bema"))
		t.Fatalf("Bema failed to start: %v", err)
	}

	// In a real e2e test we would run a client here.
	// For now, we just verify it starts up.
}
