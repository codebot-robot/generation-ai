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
	"context"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// scriptedRouter answers NewSession with errs in turn, then a session.
type scriptedRouter struct {
	pb.EngineClient
	errs  []error
	calls int
}

func (r *scriptedRouter) NewSession(context.Context, *pb.NewSessionRequest, ...grpc.CallOption) (*pb.NewSessionResponse, error) {
	r.calls++
	if r.calls <= len(r.errs) {
		return nil, r.errs[r.calls-1]
	}
	return &pb.NewSessionResponse{SessionId: "k:s1"}, nil
}

func (r *scriptedRouter) Chat(context.Context, *pb.ChatRequest, ...grpc.CallOption) (*pb.ChatResponse, error) {
	return &pb.ChatResponse{Text: "yes"}, nil
}

func TestAskByName_RetriesOnlyWhileLoading(t *testing.T) {
	loadPoll = time.Millisecond
	transient := []error{
		status.Error(codes.FailedPrecondition, "model still loading"),
		status.Error(codes.FailedPrecondition, "org/M is being reloaded (the executor no longer holds it)"),
		status.Error(codes.Unavailable, "org/M failed to load on its executor; the next request loads it again: boom"),
	}
	r := &scriptedRouter{errs: transient}
	if err := askByName(t.Context(), r, "org/M", "", "hi", 4); err != nil {
		t.Fatalf("transient states should be waited out: %v", err)
	}
	if r.calls != len(transient)+1 {
		t.Errorf("NewSession calls = %d, want %d", r.calls, len(transient)+1)
	}
	for _, final := range []error{
		status.Error(codes.FailedPrecondition, "models by name need modelstore"),
		status.Error(codes.FailedPrecondition, "model org/M@main is not registered: call RegisterModel"),
		status.Error(codes.FailedPrecondition, "no executor for this model; call LoadModel"),
		status.Error(codes.Internal, "boom"),
	} {
		r := &scriptedRouter{errs: []error{final, final}}
		if err := askByName(t.Context(), r, "org/M", "", "hi", 4); err == nil || r.calls != 1 {
			t.Errorf("%v: should fail on the first answer (calls %d, err %v)", final, r.calls, err)
		}
	}
}
