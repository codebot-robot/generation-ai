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
	"testing"

	"github.com/gke-labs/gke-labs-infra/ktesting/e2e"
)

type Harness struct {
	*e2e.Harness
}

func NewHarness(t *testing.T, clusterName string) *Harness {
	return &Harness{
		Harness: e2e.NewHarness(t, clusterName),
	}
}

func (h *Harness) InstallVPA(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "vpa")
	if err != nil {
		t.Fatalf("Failed to create temp dir for VPA: %v", err)
	}
	defer os.RemoveAll(tempDir)

	cmd := exec.Command("git", "clone", "https://github.com/kubernetes/autoscaler.git")
	cmd.Dir = tempDir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Failed to clone autoscaler: %v\nOutput: %s", err, string(out))
	}

	cmd = exec.Command("./hack/vpa-up.sh")
	cmd.Dir = filepath.Join(tempDir, "autoscaler", "vertical-pod-autoscaler")
	cmd.Env = append(os.Environ(), "FEATURE_GATES=InPlaceOrRecreate=true")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("Failed to install VPA: %v\nOutput: %s", err, string(out))
	} else {
		// Configure VPA to be more aggressive and use a smaller buffer
		exec.Command("kubectl", "patch", "deployment", "vpa-recommender", "-n", "kube-system", "--type=json", "-p", `[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--recommendation-margin-fraction=0.05"}]`).Run()
		exec.Command("kubectl", "patch", "deployment", "vpa-updater", "-n", "kube-system", "--type=json", "-p", `[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--pod-update-threshold=0.05"}]`).Run()
		exec.Command("kubectl", "patch", "deployment", "vpa-updater", "-n", "kube-system", "--type=json", "-p", `[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--min-replicas=1"}]`).Run()

		// Wait for VPA recommender to be ready
		cmd = exec.Command("kubectl", "rollout", "status", "deployment/vpa-recommender", "-n", "kube-system", "--timeout=2m")
		cmd.CombinedOutput()
		cmd = exec.Command("kubectl", "rollout", "status", "deployment/vpa-updater", "-n", "kube-system", "--timeout=2m")
		cmd.CombinedOutput()
	}
}
