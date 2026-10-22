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

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// A model's plan is the switches the coordinator sends with every load
// of the model (LoadModelRequest.plan, #394): RegisterModel sets it, and
// it is kept in modelstore so it outlives the router.
const planArtifactType = "application/vnd.gke-labs.vxpu.plan.v1+json"

func planRepository(model string) string { return "vxpu-plans/" + statsKey(model) }

// putPlan replaces model's plan.
func (s *Server) putPlan(ctx context.Context, model string, plan map[string]string) error {
	data, err := json.Marshal(plan)
	if err != nil {
		return err
	}
	name := planRepository(model)
	digest, err := s.weightStore.PutBlob(ctx, name, data)
	if err != nil {
		return err
	}
	return s.weightStore.PutManifest(ctx, name, "latest", &weightstore.Manifest{
		SchemaVersion: 2, MediaType: "application/vnd.oci.image.manifest.v1+json",
		ArtifactType: planArtifactType, Config: weightstore.EmptyConfig,
		Layers: []weightstore.Layer{{MediaType: weightstore.MediaTypeFile, Digest: digest, Size: int64(len(data)),
			Annotations: map[string]string{"org.opencontainers.image.title": "plan.json"}}},
		Annotations: map[string]string{"vnd.gke-labs.vxpu.model": model},
	})
}

// getPlan returns model's plan; none was set is an empty plan.
func (s *Server) getPlan(ctx context.Context, model string) (map[string]string, error) {
	data, err := s.weightStore.FetchFile(ctx, planRepository(model), "latest", "plan.json")
	if errors.Is(err, weightstore.ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var plan map[string]string
	if err := json.Unmarshal(data, &plan); err != nil {
		return nil, fmt.Errorf("plan of %s: %w", model, err)
	}
	return plan, nil
}

// hintsKey is a plan as one comparable string (encoding/json sorts the
// keys); no plan is "".
func hintsKey(plan map[string]string) string {
	if len(plan) == 0 {
		return ""
	}
	data, err := json.Marshal(plan)
	if err != nil { // a map of strings always encodes
		return fmt.Sprint(plan)
	}
	return string(data)
}

// setHints records model's registered plan: always when it was just
// stored, otherwise (read for a load) only if none is known, so a load
// that read the plan before a newer one was stored cannot overwrite it.
func (s *Server) setHints(model string, plan map[string]string, stored bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hints == nil {
		s.hints = map[string]string{}
	}
	if _, known := s.hints[statsKey(model)]; stored || !known {
		s.hints[statsKey(model)] = hintsKey(plan)
	}
}

// planSwitches names a plan's switches, for logs: a value can be a whole
// inline placement.
func planSwitches(plan map[string]string) []string {
	names := make([]string, 0, len(plan))
	for name := range plan {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
