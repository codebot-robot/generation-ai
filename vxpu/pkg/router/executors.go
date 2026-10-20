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
	"fmt"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
)

// executorInfo is what the coordinator knows about one executor (#394):
// what it runs on, as it described itself, and the catalog machine that
// matches (nil: none does). The planner reasons over it.
type executorInfo struct {
	Endpoint string
	Probe    machines.Probe
	Machine  *machines.Machine
}

// waitForExecutor waits until the executor at endpoint answers, and
// returns what it runs on: it asks it to describe itself (once per
// endpoint) until it does. An executor that cannot describe itself (an
// older image) is ready but unknown: nil.
func (s *Server) waitForExecutor(ctx context.Context, client pb.ExecutorClient, endpoint string, attempts int) *executorInfo {
	s.mu.RLock()
	info, known := s.executors[endpoint]
	s.mu.RUnlock()
	if known {
		return &info
	}
	log := klog.FromContext(ctx)
	var resp *pb.DescribeResponse
	var err error
	for i := 0; i < attempts; i++ {
		callCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		resp, err = client.Describe(callCtx, &pb.DescribeRequest{})
		cancel()
		if err == nil || status.Code(err) == codes.Unimplemented || ctx.Err() != nil {
			break
		}
		if i+1 < attempts {
			log.Info("Waiting for the executor to answer", "endpoint", endpoint, "attempt", i, "error", err.Error())
			time.Sleep(2 * time.Second)
		}
	}
	if err != nil {
		log.Info("Executor did not describe itself", "endpoint", endpoint, "error", err.Error())
		return nil
	}
	var probe machines.Probe
	if err := json.Unmarshal([]byte(resp.HardwareJson), &probe); err != nil {
		log.Info("Executor described itself unreadably", "endpoint", endpoint, "error", err.Error())
		return nil
	}
	info = executorInfo{Endpoint: endpoint, Probe: probe, Machine: machines.Match(probe)}
	s.mu.Lock()
	if s.executors == nil {
		s.executors = map[string]executorInfo{}
	}
	s.executors[endpoint] = info
	s.mu.Unlock()
	machine := "not in the catalog"
	if info.Machine != nil {
		machine = info.Machine.Name
	}
	log.Info("Executor described", "endpoint", endpoint, "machine", machine, "cpu", probe.CPU,
		"cpu_threads", probe.CPUThreads, "ram_bytes", probe.RAMBytes, "gpu", probe.GPU, "gpu_count", probe.GPUCount)
	return &info
}

// freeExecutors describes the external executors the planner may give
// the experts of the engine at engine to: not engine itself, holding no
// model, and serving no other engine's experts. One that does not answer
// within describeWait is left out. With no engine address, none.
func (s *Server) freeExecutors(ctx context.Context, engine string) []planner.Executor {
	if engine == "" {
		return nil
	}
	var free []planner.Executor
	for _, addr := range s.externalExecutors {
		s.mu.RLock()
		_, busy := s.externalModel[addr]
		holder, serving := s.expertHolders[addr]
		info, known := s.executors[addr]
		s.mu.RUnlock()
		if addr == engine || busy || (serving && holder != engine) {
			continue
		}
		if !known {
			conn, err := dialExecutor(addr)
			if err != nil {
				continue
			}
			describeCtx, cancel := context.WithTimeout(ctx, describeWait)
			described := s.waitForExecutor(describeCtx, pb.NewExecutorClient(conn), addr, 1)
			cancel()
			closeConn(ctx, conn)
			if described == nil {
				continue
			}
			info = *described
		}
		free = append(free, planner.Executor{Address: addr, Probe: info.Probe})
	}
	return free
}

// describeWait is how long planning waits for an executor it has not
// heard from: the load it plans is waiting too.
const describeWait = 5 * time.Second

// claimHelper books the executor at addr for the experts of the engine at
// engine, before it is asked to load them, so two loads planning at once
// cannot both take it; false if another engine holds it.
func (s *Server) claimHelper(addr, engine string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if holder, ok := s.expertHolders[addr]; ok && holder != engine {
		return false
	}
	if s.expertHolders == nil {
		s.expertHolders = map[string]string{}
	}
	s.expertHolders[addr] = engine
	return true
}

// releaseHelpersLocked frees the executors holding experts for the engine
// at engine (re-planned, evicted or replaced). The caller holds s.mu.
func (s *Server) releaseHelpersLocked(engine string) {
	for addr, holder := range s.expertHolders {
		if holder == engine {
			delete(s.expertHolders, addr)
		}
	}
}

// serveExperts asks the executor at addr to serve layers of the artifact
// to the mailboxes of the engine at engine, and returns where (host:port).
// The executor is claimed for that engine first, and freed if it fails.
func (s *Server) serveExperts(ctx context.Context, engine, addr string, manifest, binding []byte, layers []int) (string, error) {
	if !s.claimHelper(addr, engine) {
		return "", fmt.Errorf("%s serves another engine's experts", addr)
	}
	served := false
	defer func() {
		if !served {
			s.mu.Lock()
			if s.expertHolders[addr] == engine {
				delete(s.expertHolders, addr)
			}
			s.mu.Unlock()
		}
	}()
	conn, err := dialExecutor(addr)
	if err != nil {
		return "", err
	}
	defer closeConn(ctx, conn)
	req := &pb.ServeExpertsRequest{ManifestJson: string(manifest), BindingJson: string(binding)}
	for _, layer := range layers {
		req.Layers = append(req.Layers, int32(layer))
	}
	callCtx, cancel := context.WithTimeout(ctx, 30*time.Minute) // it loads the banks first
	defer cancel()
	resp, err := pb.NewExecutorClient(conn).ServeExperts(callCtx, req)
	if err != nil {
		return "", err
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return "", err
	}
	served = true
	return net.JoinHostPort(host, strconv.Itoa(int(resp.Port))), nil
}

// dialExecutor connects to the executor at addr, with room for an
// artifact's graphs in one message.
func dialExecutor(addr string) (*grpc.ClientConn, error) {
	maxMsgSize := 128 * 1024 * 1024
	return grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize), grpc.MaxCallSendMsgSize(maxMsgSize)))
}
