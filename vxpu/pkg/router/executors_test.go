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

package router

import "testing"

// An executor holding one engine's experts is that engine's until the
// engine is re-planned, forgotten or replaced: another engine planning
// at the same time cannot take it, and it is not stranded afterwards.
func TestHelpers_ClaimedAndReleased(t *testing.T) {
	s := &Server{
		externalExecutors: []string{"helper:1"},
		executors:         map[string]executorInfo{"helper:1": {}},
		externalModel:     map[string]string{},
		modelEndpoint:     map[string]string{},
	}
	free := func(engine string) int { return len(s.freeExecutors(t.Context(), engine)) }
	if !s.claimHelper("helper:1", "a:1") {
		t.Fatal("a free executor could not be claimed")
	}
	if s.claimHelper("helper:1", "b:1") || free("b:1") != 0 {
		t.Error("an executor holding a's experts was offered to b")
	}
	if free("a:1") != 1 {
		t.Error("a's own helper is not offered when a is planned again")
	}
	s.forgetModel("model-a", "a:1")
	if free("b:1") != 1 || !s.claimHelper("helper:1", "b:1") {
		t.Error("forgetting a did not free its helper")
	}
	s.externalModel["b:1"] = "model-b"
	s.placedExternally("model-c", "b:1")
	if !s.claimHelper("helper:1", "c:1") {
		t.Error("replacing b's model did not free b's helper")
	}
}
