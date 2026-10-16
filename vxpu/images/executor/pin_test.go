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

package executor

import (
	"os"
	"os/exec"
	"regexp"
	"strings"
	"testing"
)

// The executor image installs transformers from the forked submodule
// when the build context has it checked out, and otherwise clones the
// commit named in the Dockerfile. The two must name the same commit, or
// a CI build (no submodule checkout) and a local build would run
// different code.
func TestTransformersPinMatchesSubmodule(t *testing.T) {
	dockerfile, err := os.ReadFile("Dockerfile")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^ARG TRANSFORMERS_COMMIT=([0-9a-f]{40})$`).FindSubmatch(dockerfile)
	if m == nil {
		t.Fatal("Dockerfile has no ARG TRANSFORMERS_COMMIT=<sha>")
	}
	pinned := string(m[1])

	// The gitlink is in the tree whether or not the submodule is
	// checked out.
	// <rev>:<path> is relative to the repository root unless the path
	// starts with ./ or ../; the test runs in vxpu/images/executor.
	out, err := exec.Command("git", "rev-parse", "HEAD:./../../third_party/transformers").Output()
	if err != nil {
		t.Skipf("not in a git checkout with the submodule gitlink: %v", err)
	}
	gitlink := strings.TrimSpace(string(out))
	if gitlink != pinned {
		t.Fatalf("Dockerfile pins transformers %s but the submodule is at %s; update ARG TRANSFORMERS_COMMIT", pinned, gitlink)
	}
}

func TestDeepseekPinMatchesSubmodule(t *testing.T) {
	dockerfile, err := os.ReadFile("Dockerfile")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^ARG DEEPSEEK_V41_COMMIT=([0-9a-f]{40})$`).FindSubmatch(dockerfile)
	if m == nil {
		t.Fatal("Dockerfile has no ARG DEEPSEEK_V41_COMMIT=<sha>")
	}
	pinned := string(m[1])
	out, err := exec.Command("git", "rev-parse", "HEAD:./../../third_party/deepseek_v41").Output()
	if err != nil {
		t.Skipf("not in a git checkout with the submodule gitlink: %v", err)
	}
	gitlink := strings.TrimSpace(string(out))
	if gitlink != pinned {
		t.Fatalf("Dockerfile pins deepseek_v41 %s but the submodule is at %s; update ARG DEEPSEEK_V41_COMMIT", pinned, gitlink)
	}
}
