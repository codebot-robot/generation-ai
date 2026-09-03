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

func TestE2E(t *testing.T) {
	if os.Getenv("RUN_E2E") == "" {
		t.Skip("Skipping E2E test; RUN_E2E not set")
	}

	h := NewHarness(t, "vxpu-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	artifactDir := filepath.Join(gitRoot, "vxpu/tests/e2e/test_artifact")
	vxpuBin := filepath.Join(gitRoot, "vxpu/tests/e2e/vxpu_bin")

	// Build the executor image
	h.DockerBuild("vxpu-executor:e2e", filepath.Join(gitRoot, "vxpu/images/executor/Dockerfile"), gitRoot)

	// Load the executor image into Kind
	h.KindLoad("vxpu-executor:e2e")

	// Export the model using the built container image to guarantee environment consistency
	h.t.Logf("Exporting model to %s", artifactDir)
	h.RunCommand("docker", "run", "--name", "exporter-temp", "--entrypoint", "python", "vxpu-executor:e2e", "-m", "vxpu.export", "trl-internal-testing/tiny-random-LlamaForCausalLM", "-o", "/tmp/test_artifact")
	h.RunCommand("mkdir", "-p", artifactDir)
	h.RunCommand("docker", "cp", "exporter-temp:/tmp/test_artifact/.", artifactDir)
	h.RunCommand("docker", "rm", "exporter-temp")

	// Build the vxpu Go CLI binary
	h.t.Logf("Building vxpu CLI binary to %s", vxpuBin)
	h.RunCommand("go", "build", "-o", vxpuBin, filepath.Join(gitRoot, "vxpu/cmd/vxpu"))

	// Custom CPU-friendly vxpu-executor Pod manifest suitable for CPU-only Kind nodes
	vxpuExecutorYaml := `apiVersion: v1
kind: Pod
metadata:
  name: vxpu-executor
  labels:
    app: vxpu-executor
spec:
  containers:
    - name: executor
      image: vxpu-executor:e2e
      imagePullPolicy: Never
      ports:
        - containerPort: 50051
      readinessProbe:
        tcpSocket:
          port: 50051
        initialDelaySeconds: 3
        periodSeconds: 2
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          memory: "2Gi"
  restartPolicy: Never`

	// Ensure clean slate
	h.DeletePod("vxpu-executor", "default")

	// Deploy the CPU-friendly executor pod
	h.KubectlApplyContent("vxpu-executor", vxpuExecutorYaml)

	// Wait for executor pod to be ready
	if err := h.WaitForPodReady("vxpu-executor", "default", 5*time.Minute); err != nil {
		fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("default"))
		t.Fatalf("vxpu-executor pod failed to become ready: %v", err)
	}

	// Run vxpu CLI ask command to load the model, create a session, and generate responses
	h.t.Log("Running vxpu ask command")
	cmd := exec.Command(vxpuBin, "ask", "--artifact", artifactDir, "Is the sky blue?")
	out, err := cmd.CombinedOutput()

	// Get and print executor logs
	podLogs := h.GetPodLogs("app=vxpu-executor", "default")
	fmt.Fprintf(os.Stderr, "vxpu-executor logs:\n%s\n", podLogs)
	t.Logf("vxpu-executor logs:\n%s", podLogs)

	if err != nil {
		t.Fatalf("vxpu ask failed: %v\nOutput: %s", err, string(out))
	}

	outputStr := string(out)
	fmt.Fprintf(os.Stderr, "vxpu ask output:\n%s\n", outputStr)
	t.Logf("vxpu ask output:\n%s", outputStr)

	if !strings.Contains(outputStr, "Is the sky blue?") {
		t.Error("CLI output does not contain the prompt")
	}
}
