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

package main

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestSanitizeModelID(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"HuggingFaceTB/SmolLM2-135M-Instruct", "huggingfacetb-smollm2-135m-instruct"},
		{"deepseek-ai/DeepSeek-R1", "deepseek-ai-deepseek-r1"},
		{"google/gemma-2-2b-it", "google-gemma-2-2b-it"},
		{"Some_Very_Long_Name_With_Underscores_And_Slashes/Here", "some-very-long-name-with-underscores-and-slashes-here"},
	}

	for _, tc := range tests {
		actual := sanitizeModelID(tc.input)
		if actual != tc.expected {
			t.Errorf("sanitizeModelID(%q) = %q, expected %q", tc.input, actual, tc.expected)
		}
	}
}

func TestGetOrStartPod_ExistingRunningPod(t *testing.T) {
	ctx := t.Context()
	modelID := "HuggingFaceTB/SmolLM2-135M-Instruct"
	sanitized := sanitizeModelID(modelID)
	podName := "manifest-executor-" + sanitized
	namespace := "test-ns"

	existingPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "10.244.1.100",
		},
	}

	clientset := fake.NewSimpleClientset(existingPod)

	s := &server{
		clientset: clientset,
		namespace: namespace,
		imageName: "test-image",
	}

	ip, err := s.getOrStartPod(ctx, modelID)
	if err != nil {
		t.Fatalf("getOrStartPod failed: %v", err)
	}

	expectedIP := "10.244.1.100"
	if ip != expectedIP {
		t.Errorf("Expected IP %q, got %q", expectedIP, ip)
	}
}
