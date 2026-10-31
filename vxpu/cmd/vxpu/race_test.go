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
	"reflect"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/capacity"
)

func TestParseZones(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"us-central1-b,us-central1-c,us-east4-b", []string{"us-central1-b", "us-central1-c", "us-east4-b"}},
		{"us-central1-b us-central1-c us-east4-b", []string{"us-central1-b", "us-central1-c", "us-east4-b"}},
		{" us-central1-b , us-central1-c ", []string{"us-central1-b", "us-central1-c"}},
		{"", nil},
	}
	for _, tc := range tests {
		got := parseZones(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseZones(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestParseLabels(t *testing.T) {
	got := parseLabels("purpose=vxpu-hero,env=test")
	want := map[string]string{
		"purpose": "vxpu-hero",
		"env":     "test",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseLabels() = %v, want %v", got, want)
	}
}

func TestParseAccelerator(t *testing.T) {
	tests := []struct {
		input string
		want  capacity.AcceleratorSpec
	}{
		{"type=nvidia-rtx-pro-6000,count=4", capacity.AcceleratorSpec{Type: "nvidia-rtx-pro-6000", Count: 4}},
		{"nvidia-rtx-pro-6000:2", capacity.AcceleratorSpec{Type: "nvidia-rtx-pro-6000", Count: 2}},
		{"nvidia-l4", capacity.AcceleratorSpec{Type: "nvidia-l4", Count: 1}},
	}
	for _, tc := range tests {
		got, err := parseAccelerator(tc.input)
		if err != nil {
			t.Fatalf("parseAccelerator(%q) failed: %v", tc.input, err)
		}
		if got != tc.want {
			t.Errorf("parseAccelerator(%q) = %+v, want %+v", tc.input, got, tc.want)
		}
	}
}
