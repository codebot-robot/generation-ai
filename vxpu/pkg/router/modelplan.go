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
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
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

// modelPlan is the plan the coordinator chose for a model's last load,
// and why (ExplainModel).
type modelPlan struct {
	Model    string            `json:"model"`
	Machine  string            `json:"machine,omitempty"` // the executor's, as it described itself
	Time     string            `json:"time"`
	Switches map[string]string `json:"switches"`        // what was sent
	Hints    map[string]string `json:"hints,omitempty"` // the registered plan's switches, which win
	// the layers other executors serve to the engine's mailboxes, by
	// where they answer
	ExpertExecutors map[string][]int `json:"expert_executors,omitempty"`
	Reasons         []string         `json:"reasons"`
}

// planLoad chooses the switches a load of model sends: the planner's,
// from the artifact, the executor's machine and the model's recorded
// selections, with the registered plan's switches (hints) over them.
func (s *Server) planLoad(ctx context.Context, model string, manifest, binding []byte, hints map[string]string,
	info *executorInfo) map[string]string {
	log := klog.FromContext(ctx)
	var probe *machines.Probe
	mp := modelPlan{Model: model, Time: time.Now().UTC().Format(time.RFC3339), Hints: hints, Switches: map[string]string{}}
	if info != nil {
		probe = &info.Probe
		if info.Machine != nil {
			mp.Machine = info.Machine.Name
		}
	}
	if s.stats != nil { // the requests still buffered count too
		if err := s.stats.flush(ctx, model); err != nil {
			log.Error(err, "flushing statistics before planning", "model", model)
		}
	}
	artifact, err := planner.ReadArtifact(manifest, binding)
	if err != nil {
		mp.Reasons = append(mp.Reasons, fmt.Sprintf("planner: could not read the artifact (%v); the registered plan only", err))
	} else {
		counts, err := s.expertCounts(ctx, model)
		if err != nil {
			mp.Reasons = append(mp.Reasons, fmt.Sprintf("planner: could not read the statistics (%v)", err))
		}
		engine := ""
		if info != nil {
			engine = info.Endpoint
		}
		s.mu.Lock()
		s.releaseHelpersLocked(engine) // its experts are placed again
		s.mu.Unlock()
		p := planner.Choose(artifact, probe, s.freeExecutors(ctx, engine), counts)
		mp.Switches, mp.Reasons = p.Switches, append(mp.Reasons, p.Reasons...)
		// the other executors load their layers and answer on a port each;
		// the engine's mailboxes are wired to them
		var remotes []string
		for _, addr := range sortedKeys(p.ExpertExecutors) {
			layers := p.ExpertExecutors[addr]
			remote, err := s.serveExperts(ctx, engine, addr, manifest, binding, layers)
			if err != nil {
				mp.Reasons = append(mp.Reasons, fmt.Sprintf("experts: %s could not serve layers %v (%v): the engine holds them", addr, layers, err))
				continue
			}
			remotes = append(remotes, remote)
			if mp.ExpertExecutors == nil {
				mp.ExpertExecutors = map[string][]int{}
			}
			mp.ExpertExecutors[remote] = layers
		}
		if len(remotes) > 0 {
			mp.Switches["VXPU_EXPERT_REMOTE"] = strings.Join(remotes, ",")
		} else {
			delete(mp.Switches, "VXPU_EXPERT_MAILBOXES") // only for the other executors
		}
	}
	for k, v := range hints {
		if _, chosen := mp.Switches[k]; chosen {
			mp.Reasons = append(mp.Reasons, fmt.Sprintf("%s: the registered plan's value, over the planner's", k))
		}
		mp.Switches[k] = v
	}
	s.mu.Lock()
	if s.plans == nil {
		s.plans = map[string]modelPlan{}
	}
	s.plans[statsKey(model)] = mp
	s.mu.Unlock()
	log.Info("Planned the load", "model", model, "machine", mp.Machine, "switches", len(mp.Switches), "reasons", mp.Reasons)
	return mp.Switches
}

// ExplainModel returns the plan the coordinator chose for a model's last
// load, and why.
func (s *Server) ExplainModel(ctx context.Context, req *pb.ExplainModelRequest) (*pb.ExplainModelResponse, error) {
	s.mu.RLock()
	mp, ok := s.plans[statsKey(req.Model)]
	s.mu.RUnlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "%s has not been loaded by name since the router started", req.Model)
	}
	data, err := json.MarshalIndent(mp, "", "  ")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}
	return &pb.ExplainModelResponse{PlanJson: string(data)}, nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// routerOwnedSwitches are the plan switches only the router sets: which
// expert servers an engine trusts is the router's to wire (the executors
// it asked to serve them), never a client's to name.
var routerOwnedSwitches = []string{"VXPU_EXPERT_REMOTE"}
