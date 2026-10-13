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

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ModelSpec defines the desired state of Model
type ModelSpec struct {
	// Files is a list of files that make up the model.
	Files []File `json:"files,omitempty"`
	// Source records where the files were imported from, when they
	// were imported rather than uploaded.
	Source *ModelSource `json:"source,omitempty"`
}

type File struct {
	// Path is the relative path of the file within the model.
	Path string `json:"path"`
	// SHA256 is the SHA256 hash of the file content.
	SHA256 string `json:"sha256"`
	// Size is the file size in bytes, when known.
	Size int64 `json:"size,omitempty"`
}

// ModelSource is a Hugging Face Hub repository at a revision.
type ModelSource struct {
	// Repo is the Hub repository id, e.g. Qwen/Qwen3-8B-FP8.
	Repo string `json:"repo"`
	// Revision is the revision that was asked for (a branch, tag or commit).
	Revision string `json:"revision,omitempty"`
	// CommitSHA is the commit the revision resolved to at import time.
	CommitSHA string `json:"commitSha,omitempty"`
}

// Model phases.
const (
	PhaseImporting = "Importing"
	PhaseReady     = "Ready"
	PhaseFailed    = "Failed"
)

// ModelStatus defines the observed state of Model
type ModelStatus struct {
	// Phase is Importing, Ready or Failed.
	Phase string `json:"phase,omitempty"`
	// FilesTotal and FilesImported count the files of an import.
	FilesTotal    int `json:"filesTotal,omitempty"`
	FilesImported int `json:"filesImported,omitempty"`
	// BytesImported counts the bytes written to the store by this import.
	BytesImported int64 `json:"bytesImported,omitempty"`
	// Message explains a Failed phase.
	Message string `json:"message,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Files",type=integer,JSONPath=`.status.filesImported`
// +kubebuilder:printcolumn:name="Bytes",type=integer,JSONPath=`.status.bytesImported`
// +kubebuilder:printcolumn:name="Source",type=string,JSONPath=`.spec.source.repo`

// Model is the Schema for the models API
type Model struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ModelSpec   `json:"spec,omitempty"`
	Status ModelStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ModelList contains a list of Model
type ModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Model `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Model{}, &ModelList{})
}
