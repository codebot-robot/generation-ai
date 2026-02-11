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
}

func NewHarness(t *testing.T, clusterName string) *Harness {
	return &Harness{
		ClusterName: clusterName,
		t:           t,
	}
}

func (h *Harness) Setup() {
	h.t.Helper()
	// Check if cluster exists
	cmd := exec.Command("kind", "get", "clusters")
	out, err := cmd.Output()
	if err == nil && strings.Contains(string(out), h.ClusterName) {
		h.t.Logf("Cluster %s already exists", h.ClusterName)
		return
	}

	h.t.Logf("Creating cluster %s", h.ClusterName)
	cmd = exec.Command("kind", "create", "cluster", "--name", h.ClusterName)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Fatalf("Failed to create cluster: %v\nOutput: %s", err, out)
	}

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

func (h *Harness) KubectlApplyContent(content string) {
	h.t.Helper()
	h.t.Logf("Applying manifest content")
	cmd := exec.Command("kubectl", "apply", "-f", "-")
	cmd.Stdin = bytes.NewBufferString(content)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Fatalf("Failed to apply content: %v\nOutput: %s", err, out)
	}
}

func (h *Harness) WaitForDeployment(name string, timeout time.Duration) {
	h.t.Helper()
	h.t.Logf("Waiting for deployment %s", name)
	h.RunCommand("kubectl", "rollout", "status", "deployment/"+name, "--timeout="+timeout.String())
}

func (h *Harness) DeleteDeployment(name string) {
	h.t.Helper()
	// Ignore errors if deployment doesn't exist
	exec.Command("kubectl", "delete", "deployment", name, "--ignore-not-found").Run()
}

func (h *Harness) DeleteService(name string) {
	h.t.Helper()
	// Ignore errors if service doesn't exist
	exec.Command("kubectl", "delete", "service", name, "--ignore-not-found").Run()
}

func (h *Harness) DeletePod(name string) {
	h.t.Helper()
	// Ignore errors if pod doesn't exist
	exec.Command("kubectl", "delete", "pod", name, "--ignore-not-found").Run()
}

func (h *Harness) DeleteJob(name string) {
	h.t.Helper()
	// Ignore errors if job doesn't exist
	exec.Command("kubectl", "delete", "job", name, "--ignore-not-found").Run()
}

func (h *Harness) GetPodLogs(labelSelector string) string {
	h.t.Helper()
	out, err := exec.Command("kubectl", "logs", "-l", labelSelector).CombinedOutput()
	if err != nil {
		h.t.Fatalf("Failed to get logs for selector %s: %v", labelSelector, err)
	}
	return string(out)
}

func (h *Harness) WaitForJobSuccess(name string, timeout time.Duration) error {
	h.t.Helper()
	h.t.Logf("Waiting for job %s to succeed (timeout: %s)", name, timeout)
	start := time.Now()
	for {
		if time.Since(start) > timeout {
			return fmt.Errorf("timed out waiting for job %s to succeed after %s", name, timeout)
		}
		cmd := exec.Command("kubectl", "get", "job", name, "-o", "jsonpath={.status.succeeded}")
		out, err := cmd.Output()
		if err == nil && string(out) == "1" {
			h.t.Logf("Job %s succeeded", name)
			return nil
		}

		cmd = exec.Command("kubectl", "get", "job", name, "-o", "jsonpath={.status.failed}")
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
