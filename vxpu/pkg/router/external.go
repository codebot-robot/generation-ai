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
	stderrors "errors"
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// External executors (#357): executor servers on machines the router does
// not create (GPU VMs launched with gcloud, Macs), each running
// `python -m vxpu.server`. GPUs are easier to obtain as VMs than as
// Kubernetes nodes, an executor machine is not shared with other
// workloads, and its local disk is its weight cache. The router only
// dials them: models loaded by name are placed on one, and sessions are
// routed to the executor holding their model. Executor pods remain the
// path for artifacts shipped with LoadModel.

// UseExternalExecutors sets the external executors' addresses (host:port).
func (s *Server) UseExternalExecutors(addrs []string) {
	s.externalExecutors = append([]string(nil), addrs...)
}

// pickExternalExecutor chooses where model modelKey goes: the executor
// that already holds it, else one holding nothing, else the first (an
// executor holds one model; loading another replaces it).
func (s *Server) pickExternalExecutor(modelKey string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if addr, ok := s.modelEndpoint[modelKey]; ok {
		return addr
	}
	for _, addr := range s.externalExecutors {
		if _, busy := s.externalModel[addr]; !busy {
			return addr
		}
	}
	return s.externalExecutors[0]
}

// placedExternally records that the executor at addr now holds modelKey
// (and no longer whatever it held before).
func (s *Server) placedExternally(modelKey, addr string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if previous, ok := s.externalModel[addr]; ok && previous != modelKey {
		delete(s.modelEndpoint, previous)
		for tag, key := range s.loadedTags {
			if key == previous {
				delete(s.loadedTags, tag)
			}
		}
	}
	s.externalModel[addr] = modelKey
	s.modelEndpoint[modelKey] = addr
}

// executorEndpoint is the address of the executor serving modelKey: the
// external executor holding it, else its executor pod. It never creates
// a pod (only LoadModel and loading by name do).
func (s *Server) executorEndpoint(ctx context.Context, modelKey string) (string, error) {
	s.mu.RLock()
	addr, ok := s.modelEndpoint[modelKey]
	s.mu.RUnlock()
	if ok {
		return addr, nil
	}
	podIP, err := s.getOrStartPod(ctx, modelKey, nil)
	if stderrors.Is(err, errNoExecutor) {
		return "", status.Error(codes.FailedPrecondition, err.Error())
	}
	if err != nil {
		return "", status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}
	return fmt.Sprintf("%s:%d", podIP, s.executorPort), nil
}

// forgetModel drops what the router knows about where modelKey is loaded
// (its executor no longer holds it), so the next request loads it again.
func (s *Server) forgetModel(modelKey string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if addr, ok := s.modelEndpoint[modelKey]; ok && s.externalModel[addr] == modelKey {
		delete(s.externalModel, addr)
	}
	delete(s.modelEndpoint, modelKey)
	for tag, key := range s.loadedTags {
		if key == modelKey {
			delete(s.loadedTags, tag)
		}
	}
}
