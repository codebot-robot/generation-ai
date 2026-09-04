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

package v1alpha1

type Manifest struct {
	Format  string                  `json:"format"`
	Source  ManifestSource          `json:"source"`
	Config  map[string]any          `json:"config"`
	Files   map[string]ManifestFile `json:"files"`
	Tensors map[string]any          `json:"tensors"`
}

type ManifestSource struct {
	Repo      string `json:"repo"`
	Revision  string `json:"revision"`
	CommitSHA string `json:"commit_sha"`
}

type ManifestFile struct {
	Size   int64  `json:"size"`
	Name   string `json:"name"`
	Source string `json:"source"`
}
