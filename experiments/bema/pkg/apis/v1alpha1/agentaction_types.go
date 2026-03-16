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
	"k8s.io/apimachinery/pkg/runtime"
)

// AgentActionSpec defines the desired state of AgentAction
type AgentActionSpec struct {
	// SessionID is the ID of the session this action belongs to.
	SessionID string `json:"sessionId"`
	// Timestamp is the timestamp of the message that triggered this action.
	Timestamp metav1.Time `json:"timestamp"`
	// ToolName is the name of the tool to be executed.
	ToolName string `json:"toolName"`
	// ToolArgs are the arguments for the tool.
	ToolArgs runtime.RawExtension `json:"toolArgs,omitempty"`
}

// AgentActionStatus defines the observed state of AgentAction
type AgentActionStatus struct {
	// Status is the current status of the action.
	Status string `json:"status,omitempty"`
	// ToolOutput is the output of the tool execution.
	ToolOutput runtime.RawExtension `json:"toolOutput,omitempty"`
	// UndoData is the data needed to undo the action.
	UndoData runtime.RawExtension `json:"undoData,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// AgentAction is the Schema for the agentactions API
type AgentAction struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AgentActionSpec   `json:"spec,omitempty"`
	Status AgentActionStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AgentActionList contains a list of AgentAction
type AgentActionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AgentAction `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AgentAction{}, &AgentActionList{})
}
