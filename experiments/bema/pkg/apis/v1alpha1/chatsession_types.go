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

// ChatSessionSpec defines the desired state of ChatSession
type ChatSessionSpec struct {
	// Config is the session configuration.
	// +kubebuilder:pruning:PreserveUnknownFields
	Config runtime.RawExtension `json:"config,omitempty"`
}

// ChatSessionStatus defines the observed state of ChatSession
type ChatSessionStatus struct {
	// Status is the status of the session.
	Status string `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// ChatSession is the Schema for the chatsessions API
type ChatSession struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ChatSessionSpec   `json:"spec,omitempty"`
	Status ChatSessionStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ChatSessionList contains a list of ChatSession
type ChatSessionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ChatSession `json:"items"`
}

// ChatSessionMessageSpec defines the desired state of ChatSessionMessage
type ChatSessionMessageSpec struct {
	// SessionID is the ID of the ChatSession this message belongs to.
	SessionID string `json:"sessionId"`

	// Role is the role of the message (e.g. user, model, system, function).
	Role string `json:"role"`

	// Parts is the content of the message.
	// +kubebuilder:pruning:PreserveUnknownFields
	Parts runtime.RawExtension `json:"parts,omitempty"`

	// Timestamp is the time the message was created.
	Timestamp metav1.Time `json:"timestamp,omitempty"`

	// Index is the order of the message in the session.
	Index int32 `json:"index"`
}

// +kubebuilder:object:root=true

// ChatSessionMessage is the Schema for the chatsessionmessages API
type ChatSessionMessage struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec ChatSessionMessageSpec `json:"spec,omitempty"`
}

// +kubebuilder:object:root=true

// ChatSessionMessageList contains a list of ChatSessionMessage
type ChatSessionMessageList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ChatSessionMessage `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ChatSession{}, &ChatSessionList{})
	SchemeBuilder.Register(&ChatSessionMessage{}, &ChatSessionMessageList{})
}
