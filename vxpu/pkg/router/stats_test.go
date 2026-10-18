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
	"encoding/json"
	"net"
	"strings"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func TestStats_RecordsNamedModelRequests(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseStats(1)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, &loadingExecutor{})
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()
	s.UseExternalExecutors([]string{lis.Addr().String()})

	session, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	reply, err := s.Chat(ctx, &pb.ChatRequest{SessionId: session.SessionId, Text: "hi"})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if reply.TelemetryJson != "" {
		t.Error("the engine's telemetry must not reach a client that did not ask for it")
	}

	stats, err := s.GetModelStats(ctx, &pb.GetModelStatsRequest{Model: "org/M", Records: true})
	if err != nil {
		t.Fatalf("GetModelStats: %v", err)
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(stats.SummaryJson), &summary); err != nil {
		t.Fatalf("summary: %v", err)
	}
	if summary["requests"] != 1.0 || summary["telemetry_sampled"] != 1.0 || summary["generated_tokens"] != 10.0 {
		t.Errorf("unexpected summary: %s", stats.SummaryJson)
	}
	if byKind, _ := summary["by_kind_ms_per_step"].(map[string]any); byKind["dense"] != 20.0 {
		t.Errorf("segment times not summarised: %s", stats.SummaryJson)
	}
	experts, _ := summary["experts_decode"].(map[string]any)
	if experts["selections"] != 12.0 || experts["experts_per_layer_seen"] != 4.0 {
		t.Errorf("expert counts not summarised: %s", stats.SummaryJson)
	}
	if !strings.Contains(stats.RecordsJsonl, `"rpc":"chat"`) || !strings.Contains(stats.RecordsJsonl, `"vxpu-telemetry/1"`) ||
		!strings.Contains(stats.RecordsJsonl, `"commit":"`+testCommit+`"`) {
		t.Errorf("records lack the request, its telemetry or its commit: %s", stats.RecordsJsonl)
	}
}
