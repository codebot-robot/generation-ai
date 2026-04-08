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
	"bytes"
	"crypto/rand"
	"fmt"
	"github.com/bradfitz/gomemcache/memcache"
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
	h.KindLoad("scalingpolicy-controller:e2e")

	h.DockerBuild("test-memcache-server:e2e", filepath.Join(experimentRoot, "images/test-memcache-server/Dockerfile"), experimentRoot)
	h.KindLoad("test-memcache-server:e2e")

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
	manifest = strings.ReplaceAll(manifest, "image: scalingpolicy-controller:latest", "image: scalingpolicy-controller:e2e\n        imagePullPolicy: Never")
	h.KubectlApplyContent("scalingpolicy-manifest", manifest)

	// Install metrics-server
	tempDir, err := os.MkdirTemp("", "metrics-server")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	kustomization := `resources:
- https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml

apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
patches:
- patch: |-
    - op: add
      path: /spec/template/spec/containers/0/args/-
      value: --kubelet-insecure-tls
  target:
    group: apps
    kind: Deployment
    name: metrics-server
    namespace: kube-system
    version: v1
`
	if err := os.WriteFile(filepath.Join(tempDir, "kustomization.yaml"), []byte(kustomization), 0644); err != nil {
		t.Fatalf("Failed to write kustomization.yaml: %v", err)
	}

	cmd := exec.Command("kubectl", "apply", "-k", tempDir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Failed to install metrics-server: %v\nOutput: %s", err, out)
	}

	// Wait for metrics-server to be ready
	cmd = exec.Command("kubectl", "rollout", "status", "deployment/metrics-server", "-n", "kube-system", "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("metrics-server failed to start: %v\nOutput: %s", err, out)
	}

	if err := h.WaitForStatefulSet("scalingpolicy-controller", "kube-scalingpolicy-system", 2*time.Minute); err != nil {
		t.Fatalf("ScalingPolicy Controller failed to start: %v", err)
	}
	t.Log("ScalingPolicy controller started successfully.")

	// Deploy memcache server and ScalingPolicy
	testManifest := `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memcache-deployment
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: memcache
  template:
    metadata:
      labels:
        app: memcache
    spec:
      containers:
      - name: memcached
        image: test-memcache-server:e2e
        imagePullPolicy: Never
        ports:
        - containerPort: 11211
        resources:
          limits:
            memory: 64Mi
---
apiVersion: generation-ai.gke.io/v1alpha1
kind: ScalingPolicy
metadata:
  name: memcache-policy
  namespace: default
spec:
  target:
    kind: Deployment
    name: memcache-deployment
  inputs:
  - name: pod_memory
    metric: memory
  values:
  - path: spec.template.spec.containers[0].resources.limits.memory
    expression: "pod_memory + 64 * 1024 * 1024"
    min: 67108864 # 64MiB
    max: 536870912 # 512MiB
`
	h.KubectlApplyContent("test-resources", testManifest)

	// Wait for deployment to be ready
	cmd = exec.Command("kubectl", "rollout", "status", "deployment/memcache-deployment", "-n", "default", "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("memcache-deployment failed to start: %v\nOutput: %s", err, out)
	}

	// Port forward to memcache
	pfCmd := exec.Command("kubectl", "port-forward", "deployment/memcache-deployment", "11211:11211", "-n", "default")
	var pfOut bytes.Buffer
	pfCmd.Stdout = &pfOut
	pfCmd.Stderr = &pfOut
	if err := pfCmd.Start(); err != nil {
		t.Fatalf("Failed to start port-forward: %v", err)
	}
	defer pfCmd.Process.Kill()

	// Wait for port-forward to be ready
	time.Sleep(5 * time.Second)

	mc := memcache.New("localhost:11211")

	// Write random values
	go func() {
		val := make([]byte, 512*1024) // 512KB chunks
		for i := 0; i < 1200; i++ {
			rand.Read(val)
			key := fmt.Sprintf("key-%d", i)
			err := mc.Set(&memcache.Item{Key: key, Value: val})
			if err != nil {
				t.Logf("Failed to write to memcache: %v", err)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}()

	// Monitor memory limits
	deadline := time.Now().Add(5 * time.Minute)
	success := false
	for time.Now().Before(deadline) {
		cmd := exec.Command("kubectl", "get", "deployment", "memcache-deployment", "-n", "default", "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}")
		out, err := cmd.Output()
		if err == nil {
			limitStr := string(out)
			t.Logf("Current memory limit: %s", limitStr)
			if limitStr == "512Mi" || limitStr == "536870912" {
				success = true
				break
			}
		}
		time.Sleep(5 * time.Second)
	}
	if !success {
		t.Fatalf("Deployment memory limit did not reach 512Mi")
	}

	t.Log("ScalingPolicy successfully scaled memcache deployment memory to 512MiB.")
}
