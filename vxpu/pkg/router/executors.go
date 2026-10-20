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
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// executorInfo is what the coordinator knows about one executor (#394):
// what it runs on, as it described itself, and the catalog machine that
// matches (nil: none does). The planner reasons over it.
type executorInfo struct {
	Probe   machines.Probe
	Machine *machines.Machine
}

// waitForExecutor waits until the executor at endpoint answers, and
// returns what it runs on: it asks it to describe itself (once per
// endpoint) until it does. An executor that cannot describe itself (an
// older image) is ready but unknown: nil.
func (s *Server) waitForExecutor(ctx context.Context, client pb.ExecutorClient, endpoint string) *executorInfo {
	s.mu.RLock()
	info, known := s.executors[endpoint]
	s.mu.RUnlock()
	if known {
		return &info
	}
	log := klog.FromContext(ctx)
	var resp *pb.DescribeResponse
	var err error
	for i := 0; i < 30; i++ {
		callCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		resp, err = client.Describe(callCtx, &pb.DescribeRequest{})
		cancel()
		if err == nil || status.Code(err) == codes.Unimplemented || ctx.Err() != nil {
			break
		}
		log.Info("Waiting for the executor to answer", "endpoint", endpoint, "attempt", i, "error", err.Error())
		time.Sleep(2 * time.Second)
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
	info = executorInfo{Probe: probe, Machine: machines.Match(probe)}
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
