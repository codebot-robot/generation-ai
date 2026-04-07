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

	h := NewHarness(t, "bema-e2e")
	h.Setup()
	h.TrackNamespace("bema")
	defer func() {
		if t.Failed() {
			h.CollectArtifacts("TestE2E")
		}
	}()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/bema")

	// Build images
	h.DockerBuild("bema:e2e", filepath.Join(experimentRoot, "images/bema/Dockerfile"), experimentRoot)

	// Load images into Kind
	h.KindLoad("bema:e2e")

	// Create namespace and secret first
	h.RunCommand("kubectl", "create", "namespace", "bema")
	h.RunCommand("kubectl", "create", "secret", "generic", "bema", "--from-literal=dummy=value", "-n", "bema")

	// Apply all manifests
	k8sDir := filepath.Join(experimentRoot, "k8s")
	files, err := os.ReadDir(k8sDir)
	if err != nil {
		t.Fatalf("Failed to read k8s dir: %v", err)
	}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		path := filepath.Join(k8sDir, file.Name())
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("Failed to read manifest %s: %v", path, err)
		}
		manifest := string(b)
		if file.Name() == "manifest.yaml" {
			manifest = strings.ReplaceAll(manifest, "image: bema:latest", "image: bema:e2e\n          imagePullPolicy: Never")
		}
		if file.Name() == "cert-manager.yaml" {
			h.RunCommand("kubectl", "apply", "-f", "https://github.com/cert-manager/cert-manager/releases/download/v1.17.1/cert-manager.yaml")
			time.Sleep(30 * time.Second)
		}
		h.KubectlApplyContent(file.Name(), manifest)
	}

	// Wait for server
	if err := h.WaitForStatefulSet("bema", "bema", 5*time.Minute); err != nil {
		t.Fatalf("Bema failed to start: %v", err)
	}
}
