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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeMetricsToCSV(t *testing.T, namespace, scenario string) {
	// Give it a moment to collect final stats
	time.Sleep(5 * time.Second)
	cmd := exec.Command("kubectl", "exec", "deployment/memcache-client", "-n", namespace, "--", "curl", "-s", "http://memcache-service:8080/metrics")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("Failed to get metrics for %s: %v\nOutput: %s", scenario, err, out)
		return
	}

	hits := "0"
	misses := "0"
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "memcached_get_hits ") {
			hits = strings.TrimSpace(strings.TrimPrefix(line, "memcached_get_hits "))
		}
		if strings.HasPrefix(line, "memcached_get_misses ") {
			misses = strings.TrimSpace(strings.TrimPrefix(line, "memcached_get_misses "))
		}
	}

	artifactsDir := os.Getenv("ARTIFACTS")
	if artifactsDir == "" {
		artifactsDir = "/tmp"
	}
	csvPath := filepath.Join(artifactsDir, "metrics.csv")

	f, err := os.OpenFile(csvPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Logf("Failed to open csv file: %v", err)
		return
	}
	defer f.Close()

	statLine := fmt.Sprintf("%s,%s,%s\n", scenario, hits, misses)
	if _, err := f.WriteString(statLine); err != nil {
		t.Logf("Failed to write to csv: %v", err)
	}
}

func deployMemcacheApp(t *testing.T, h *Harness, namespace string, manifestExtras string) {
	manifest := fmt.Sprintf(`
apiVersion: v1
kind: Namespace
metadata:
  name: %[1]s
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memcache-deployment
  namespace: %[1]s
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
        - containerPort: 8080
        resources:
          limits:
            memory: 64Mi
          requests:
            memory: 64Mi
---
apiVersion: v1
kind: Service
metadata:
  name: memcache-service
  namespace: %[1]s
spec:
  selector:
    app: memcache
  ports:
  - port: 11211
    targetPort: 11211
    name: memcache
  - port: 8080
    targetPort: 8080
    name: metrics
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memcache-client
  namespace: %[1]s
spec:
  replicas: 1
  selector:
    matchLabels:
      app: memcache-client
  template:
    metadata:
      labels:
        app: memcache-client
    spec:
      containers:
      - name: client
        image: test-memcache-client:e2e
        imagePullPolicy: Never
        args:
        - "-server=memcache-service:11211"
        - "-min-ops=20"
        - "-max-ops=50"
%[2]s
`, namespace, manifestExtras)
	h.KubectlApplyContent("test-resources-"+namespace, manifest)

	// Wait for deployment to be ready
	cmd := exec.Command("kubectl", "rollout", "status", "deployment/memcache-deployment", "-n", namespace, "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("memcache-deployment failed to start: %v\nOutput: %s", err, out)
	}

	cmd = exec.Command("kubectl", "rollout", "status", "deployment/memcache-client", "-n", namespace, "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("memcache-client failed to start: %v\nOutput: %s", err, out)
	}
}

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

	h.DockerBuild("test-memcache-client:e2e", filepath.Join(experimentRoot, "images/test-memcache-client/Dockerfile"), experimentRoot)
	h.KindLoad("test-memcache-client:e2e")

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

	cmd = exec.Command("kubectl", "rollout", "status", "deployment/metrics-server", "-n", "kube-system", "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("metrics-server failed to start: %v\nOutput: %s", err, out)
	}

	// Install VPA
	cmd = exec.Command("kubectl", "apply", "-k", "github.com/kubernetes/autoscaler/vertical-pod-autoscaler/deploy/kustomize?ref=master")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("Failed to install VPA, VPA tests might fail: %v\nOutput: %s", err, out)
	} else {
		// Wait for VPA recommender to be ready
		cmd = exec.Command("kubectl", "rollout", "status", "deployment/vpa-recommender", "-n", "kube-system", "--timeout=2m")
		cmd.CombinedOutput()
		cmd = exec.Command("kubectl", "rollout", "status", "deployment/vpa-updater", "-n", "kube-system", "--timeout=2m")
		cmd.CombinedOutput()
	}

	if err := h.WaitForStatefulSet("scalingpolicy-controller", "kube-scalingpolicy-system", 2*time.Minute); err != nil {
		t.Fatalf("ScalingPolicy Controller failed to start: %v", err)
	}
	t.Log("ScalingPolicy controller started successfully.")

	// CSV Header
	artifactsDir := os.Getenv("ARTIFACTS")
	if artifactsDir == "" {
		artifactsDir = "/tmp"
	}
	csvPath := filepath.Join(artifactsDir, "metrics.csv")
	os.WriteFile(csvPath, []byte("scenario,hits,misses\n"), 0644)

	t.Run("ScalingPolicy", func(t *testing.T) {
		ns := "test-sp"
		policy := `---
apiVersion: generation-ai.gke.io/v1alpha1
kind: ScalingPolicy
metadata:
  name: memcache-policy
  namespace: test-sp
spec:
  target:
    kind: Deployment
    name: memcache-deployment
  inputs:
  - name: pod_memory
    metric: memory
  values:
  - path: spec.template.spec.containers[0].resources.limits.memory
    expression: "pod_memory + 128 * 1024 * 1024"
    min: 67108864 # 64MiB
    max: 536870912 # 512MiB
`
		deployMemcacheApp(t, h, ns, policy)

		deadline := time.Now().Add(5 * time.Minute)
		success := false
		for time.Now().Before(deadline) {
			cmd := exec.Command("sh", "-c", "kubectl get pods -l app=memcache -n "+ns+" -o jsonpath='{.items[0].spec.containers[0].resources.limits.memory}'")
			out, err := cmd.Output()
			if err == nil {
				limitStr := string(out)
				if limitStr == "512Mi" || limitStr == "536870912" {
					success = true
					break
				}
			}
			time.Sleep(5 * time.Second)
		}
		if !success {
			t.Log("Warning: Deployment memory limit did not reach 512Mi")
		}
		writeMetricsToCSV(t, ns, "ScalingPolicy")
	})

	t.Run("VPA", func(t *testing.T) {
		ns := "test-vpa"
		vpa := `---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: memcache-vpa
  namespace: test-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind:       Deployment
    name:       memcache-deployment
  updatePolicy:
    updateMode: "Auto"
`
		deployMemcacheApp(t, h, ns, vpa)

		// Let it run for 2 minutes to gather metrics and potentially scale
		time.Sleep(2 * time.Minute)
		writeMetricsToCSV(t, ns, "VPA")
	})

	t.Run("Fixed", func(t *testing.T) {
		ns := "test-fixed"
		// No extra policy or VPA
		deployMemcacheApp(t, h, ns, "")

		// Let it run for 2 minutes to gather metrics
		time.Sleep(2 * time.Minute)
		writeMetricsToCSV(t, ns, "Fixed")
	})
}
