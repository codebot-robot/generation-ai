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

// TargetRef defines the target workload.
type TargetRef struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
}

// Input defines a metric source.
type Input struct {
	Name   string `json:"name"`
	Metric string `json:"metric"`
}

// Value defines the scaling logic using CEL expression.
type Value struct {
	Path       string `json:"path"`
	Expression string `json:"expression"`
	Min        string `json:"min"`
	Max        string `json:"max"`
}

// ScalingPolicySpec defines the desired state of ScalingPolicy.
type ScalingPolicySpec struct {
	Target TargetRef `json:"target"`
	Inputs []Input   `json:"inputs"`
	Values []Value   `json:"values"`
}

// ScalingPolicyStatus defines the observed state of ScalingPolicy.
type ScalingPolicyStatus struct {
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// ScalingPolicy is the Schema for the scalingpolicies API
type ScalingPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ScalingPolicySpec   `json:"spec,omitempty"`
	Status ScalingPolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ScalingPolicyList contains a list of ScalingPolicy
type ScalingPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ScalingPolicy `json:"items"`
}
