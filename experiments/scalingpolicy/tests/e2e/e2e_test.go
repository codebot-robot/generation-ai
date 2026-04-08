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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestE2E(t *testing.T) {
	if os.Getenv("RUN_E2E") == "" {
		t.Skip("Skipping E2E test; RUN_E2E not set")
	}

	h := NewHarness(t, "scalingpolicy-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/scalingpolicy")

	// Build image
	h.DockerBuild("scalingpolicy-controller:e2e", filepath.Join(experimentRoot, "images/scalingpolicy-controller/Dockerfile"), experimentRoot)

	// Load image into Kind
	h.KindLoad("scalingpolicy-controller:e2e")

	// Apply CRD
	crdPath := filepath.Join(experimentRoot, "k8s/crds/scalingpolicy.yaml")
	b, err := os.ReadFile(crdPath)
	if err != nil {
		t.Fatalf("Failed to read CRD manifest: %v", err)
	}
	h.KubectlApplyContent("scalingpolicy-crd", string(b))

	// Apply manifest
	manifestPath := filepath.Join(experimentRoot, "k8s/manifest.yaml")
	b, err = os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("Failed to read manifest: %v", err)
	}
	manifest := string(b)
	// Replace image string to use local image
	manifest = strings.ReplaceAll(manifest, "image: scalingpolicy-controller:latest", "image: scalingpolicy-controller:e2e\n        imagePullPolicy: Never")

	h.KubectlApplyContent("scalingpolicy-manifest", manifest)

	// Wait for controller StatefulSet
	if err := h.WaitForStatefulSet("scalingpolicy-controller", "kube-scalingpolicy-system", 2*time.Minute); err != nil {
		t.Logf("Events:\n%s\n", h.GetEvents("kube-scalingpolicy-system"))
		t.Logf("Pod logs:\n%s\n", h.GetPodLogs("app=scalingpolicy-controller", "kube-scalingpolicy-system"))
		t.Fatalf("ScalingPolicy Controller failed to start: %v", err)
	}

	t.Log("ScalingPolicy controller started successfully.")

	testManifest := `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test-deployment
  template:
    metadata:
      labels:
        app: test-deployment
    spec:
      containers:
      - name: nginx
        image: nginx:latest
---
apiVersion: generation-ai.gke.io/v1alpha1
kind: ScalingPolicy
metadata:
  name: test-policy
  namespace: default
spec:
  target:
    kind: Deployment
    name: test-deployment
  inputs:
  - name: dummy
    metric: dummy
  values:
  - path: spec.replicas
    expression: "2"
    min: 1
    max: 3
`
	h.KubectlApplyContent("test-resources", testManifest)

	// Wait for deployment to have 2 replicas
	deadline := time.Now().Add(2 * time.Minute)
	success := false
	for time.Now().Before(deadline) {
		cmd := exec.Command("kubectl", "get", "deployment", "test-deployment", "-n", "default", "-o", "jsonpath={.spec.replicas}")
		out, err := cmd.Output()
		if err == nil && string(out) == "2" {
			success = true
			break
		}
		time.Sleep(2 * time.Second)
	}
	if !success {
		t.Fatalf("Deployment replicas did not scale to 2")
	}

	t.Log("ScalingPolicy successfully scaled deployment to 2 replicas.")
}
