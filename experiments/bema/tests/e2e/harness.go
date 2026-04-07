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
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"
)

type Harness struct {
	ClusterName string
	t           *testing.T
	Namespaces  []string
}

func NewHarness(t *testing.T, clusterName string) *Harness {
	return &Harness{
		ClusterName: clusterName,
		t:           t,
	}
}

func (h *Harness) TrackNamespace(namespace string) {
	h.Namespaces = append(h.Namespaces, namespace)
}

func (h *Harness) CollectArtifacts(testName string) {
	artifactsDir := os.Getenv("ARTIFACTS")
	if artifactsDir == "" {
		return
	}
	h.t.Logf("Collecting artifacts to %s", artifactsDir)

	for _, ns := range h.Namespaces {
		nsDir := filepath.Join(artifactsDir, "tests", testName, "objects", ns)
		os.MkdirAll(nsDir, 0755)

		pods, _ := exec.Command("kubectl", "get", "pods", "-n", ns).Output()
		os.WriteFile(filepath.Join(nsDir, "pods.txt"), pods, 0644)

		podsYaml, _ := exec.Command("kubectl", "get", "pods", "-n", ns, "-o", "yaml").Output()
		os.WriteFile(filepath.Join(nsDir, "pods.yaml"), podsYaml, 0644)

		logsDir := filepath.Join(artifactsDir, "tests", testName, "logs", ns)
		os.MkdirAll(logsDir, 0755)

		podList, _ := exec.Command("kubectl", "get", "pods", "-n", ns, "-o", "jsonpath={.items[*].metadata.name}").Output()
		for _, pod := range strings.Fields(string(podList)) {
			logs, _ := exec.Command("kubectl", "logs", pod, "-n", ns).Output()
			os.WriteFile(filepath.Join(logsDir, pod+".log"), logs, 0644)
		}
	}

	clusterDir := filepath.Join(artifactsDir, "tests", testName, "objects", "_cluster")
	os.MkdirAll(clusterDir, 0755)
	nodes, _ := exec.Command("kubectl", "get", "nodes").Output()
	os.WriteFile(filepath.Join(clusterDir, "nodes.txt"), nodes, 0644)
	nodesYaml, _ := exec.Command("kubectl", "get", "nodes", "-o", "yaml").Output()
	os.WriteFile(filepath.Join(clusterDir, "nodes.yaml"), nodesYaml, 0644)
}

func (h *Harness) Setup() {
	h.t.Helper()
	// Check if cluster exists
	cmd := exec.Command("kind", "get", "clusters")
	out, err := cmd.Output()
	if err == nil && strings.Contains(string(out), h.ClusterName) {
		h.t.Logf("Cluster %s already exists", h.ClusterName)
		h.RunCommand("kind", "export", "kubeconfig", "--name", h.ClusterName)
	} else {
		h.t.Logf("Creating cluster %s", h.ClusterName)
		cmd = exec.Command("kind", "create", "cluster", "--name", h.ClusterName)
		if out, err := cmd.CombinedOutput(); err != nil {
			h.t.Fatalf("Failed to create cluster: %v\nOutput: %s", err, out)
		}
	}

	// Ensure default namespace is used, avoiding issues with environment-specific defaults
	h.RunCommand("kubectl", "config", "set-context", "--current", "--namespace=default")

	h.t.Cleanup(func() {
		h.Teardown()
	})
}

func (h *Harness) Teardown() {
	h.t.Helper()
	h.t.Logf("Deleting cluster %s", h.ClusterName)
	cmd := exec.Command("kind", "delete", "cluster", "--name", h.ClusterName)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Logf("Failed to delete cluster: %v\nOutput: %s", err, out)
	}
}

func (h *Harness) GetGitRoot() string {
	h.t.Helper()
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	out, err := cmd.Output()
	if err != nil {
		h.t.Fatalf("Failed to find git root: %v", err)
	}
	return strings.TrimSpace(string(out))
}

func (h *Harness) RunCommand(name string, args ...string) {
	h.t.Helper()
	cmd := exec.Command(name, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Fatalf("Command failed: %s %v\nOutput: %s", name, args, out)
	}
}

func (h *Harness) DockerBuild(tag, dockerfile, context string) {
	h.t.Helper()
	h.t.Logf("Building docker image %s", tag)
	h.RunCommand("docker", "build", "-t", tag, "-f", dockerfile, context)
}

func (h *Harness) KindLoad(tag string) {
	h.t.Helper()
	h.t.Logf("Loading image %s into kind", tag)
	h.RunCommand("kind", "load", "docker-image", tag, "--name", h.ClusterName)
}

func (h *Harness) KubectlApplyContent(name, content string, args ...string) {
	h.t.Helper()
	snippet := content
	if len(snippet) > 100 {
		snippet = snippet[:100] + "..."
	}
	h.t.Logf("Applying manifest content for %s:\n%s", name, snippet)
	cmdArgs := append([]string{"apply", "-f", "-"}, args...)
	cmd := exec.Command("kubectl", cmdArgs...)
	cmd.Stdin = bytes.NewBufferString(content)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Fatalf("Failed to apply content for %s: %v\nOutput: %s\nFull manifest:\n%s", name, err, out, content)
	}
}

func (h *Harness) WaitForDeployment(name, namespace string, timeout time.Duration) error {
	h.t.Helper()
	h.t.Logf("Waiting for deployment %s in namespace %s", name, namespace)
	cmd := exec.Command("kubectl", "rollout", "status", "deployment/"+name, "-n", namespace, "--timeout="+timeout.String())
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("deployment %s failed to become ready: %v\nOutput: %s", name, err, out)
	}
	return nil
}

func (h *Harness) WaitForStatefulSet(name, namespace string, timeout time.Duration) error {
	h.t.Helper()
	h.t.Logf("Waiting for statefulset %s in namespace %s", name, namespace)
	cmd := exec.Command("kubectl", "rollout", "status", "statefulset/"+name, "-n", namespace, "--timeout="+timeout.String())
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("statefulset %s failed to become ready: %v\nOutput: %s", name, err, out)
	}
	return nil
}

func (h *Harness) DeleteDeployment(name, namespace string) {
	h.t.Helper()
	// Ignore errors if deployment doesn't exist
	exec.Command("kubectl", "delete", "deployment", name, "-n", namespace, "--ignore-not-found").Run()
}

func (h *Harness) DeleteStatefulSet(name, namespace string) {
	h.t.Helper()
	// Ignore errors if statefulset doesn't exist
	exec.Command("kubectl", "delete", "statefulset", name, "-n", namespace, "--ignore-not-found").Run()
}

func (h *Harness) DeleteService(name, namespace string) {
	h.t.Helper()
	// Ignore errors if service doesn't exist
	exec.Command("kubectl", "delete", "service", name, "-n", namespace, "--ignore-not-found").Run()
}

func (h *Harness) DeletePod(name, namespace string) {
	h.t.Helper()
	// Ignore errors if pod doesn't exist
	exec.Command("kubectl", "delete", "pod", name, "-n", namespace, "--ignore-not-found").Run()
}

func (h *Harness) DeleteJob(name, namespace string) {
	h.t.Helper()
	// Ignore errors if job doesn't exist
	exec.Command("kubectl", "delete", "job", name, "-n", namespace, "--ignore-not-found").Run()
}

func (h *Harness) GetPodLogs(labelSelector, namespace string) string {
	h.t.Helper()
	out, err := exec.Command("kubectl", "logs", "-l", labelSelector, "-n", namespace).CombinedOutput()
	if err != nil {
		h.t.Logf("Warning: failed to get logs for selector %s in namespace %s: %v", labelSelector, namespace, err)
		return string(out)
	}
	return string(out)
}

func (h *Harness) GetPodYaml(labelSelector, namespace string) string {
	h.t.Helper()
	out, err := exec.Command("kubectl", "get", "pod", "-l", labelSelector, "-n", namespace, "-o", "yaml").CombinedOutput()
	if err != nil {
		h.t.Logf("Warning: failed to get pod yaml for selector %s in namespace %s: %v", labelSelector, namespace, err)
		return string(out)
	}
	return string(out)
}

func (h *Harness) GetEvents(namespace string) string {
	h.t.Helper()
	cmdArgs := []string{"get", "events", "--sort-by=.lastTimestamp"}
	if namespace != "" {
		cmdArgs = append(cmdArgs, "-n", namespace)
	}
	out, err := exec.Command("kubectl", cmdArgs...).CombinedOutput()
	if err != nil {
		h.t.Logf("Warning: failed to get events: %v", err)
		return string(out)
	}
	return string(out)
}

func (h *Harness) WaitForJobSuccess(name, namespace string, timeout time.Duration) error {
	h.t.Helper()
	h.t.Logf("Waiting for job %s to succeed in namespace %s (timeout: %s)", name, namespace, timeout)
	start := time.Now()
	for {
		if time.Since(start) > timeout {
			return fmt.Errorf("timed out waiting for job %s to succeed after %s", name, timeout)
		}
		cmd := exec.Command("kubectl", "get", "job", name, "-n", namespace, "-o", "jsonpath={.status.succeeded}")
		out, err := cmd.Output()
		if err == nil && string(out) == "1" {
			h.t.Logf("Job %s succeeded", name)
			return nil
		}

		cmd = exec.Command("kubectl", "get", "job", name, "-n", namespace, "-o", "jsonpath={.status.failed}")
		out, err = cmd.Output()
		if err == nil && string(out) == "1" {
			return fmt.Errorf("job %s failed", name)
		}

		// Optional: Log status every 30 seconds
		if int(time.Since(start).Seconds())%30 < 2 {
			h.t.Logf("Still waiting for job %s...", name)
		}

		time.Sleep(2 * time.Second)
	}
}
