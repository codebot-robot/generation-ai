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

package k8sstore

import (
	"testing"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	bemav1alpha1 "github.com/gke-labs/generation-ai/experiments/bema/pkg/apis/v1alpha1"
	"google.golang.org/protobuf/types/known/structpb"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestK8sSessionStore(t *testing.T) {
	bemav1alpha1.AddToScheme(scheme.Scheme)
	c := fake.NewClientBuilder().WithScheme(scheme.Scheme).Build()
	store := NewK8sSessionStore(c, "default")

	ctx := t.Context()

	// 1. Save session
	sess := &pb.Session{
		Id:     "test-session",
		Status: "idle",
		Messages: []*pb.Message{
			{
				Role: "user",
				Parts: []*pb.Part{
					{
						Data: &pb.Part_Text{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	if err := store.SaveSession(ctx, sess); err != nil {
		t.Fatalf("SaveSession failed: %v", err)
	}

	// 2. Get session
	sess2, err := store.GetSession(ctx, "test-session")
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if sess2 == nil {
		t.Fatal("Expected session, got nil")
	}
	if sess2.Id != "test-session" {
		t.Errorf("Expected session ID 'test-session', got '%s'", sess2.Id)
	}
	if len(sess2.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(sess2.Messages))
	}
	if sess2.Messages[0].Parts[0].GetText() != "Hello" {
		t.Errorf("Expected message content 'Hello', got '%s'", sess2.Messages[0].Parts[0].GetText())
	}

	// 3. Append another message with function call
	args, _ := structpb.NewStruct(map[string]any{"arg1": "val1"})
	sess2.Messages = append(sess2.Messages, &pb.Message{
		Role: "model",
		Parts: []*pb.Part{
			{
				Data: &pb.Part_FunctionCall{
					FunctionCall: &pb.FunctionCall{
						Name: "my-func",
						Args: args,
					},
				},
			},
		},
	})

	if err := store.SaveSession(ctx, sess2); err != nil {
		t.Fatalf("SaveSession (update with func call) failed: %v", err)
	}

	// 4. Append function response
	resp, _ := structpb.NewStruct(map[string]any{"result": "ok"})
	sess2.Messages = append(sess2.Messages, &pb.Message{
		Role: "function",
		Parts: []*pb.Part{
			{
				Data: &pb.Part_FunctionResponse{
					FunctionResponse: &pb.FunctionResponse{
						Name:     "my-func",
						Response: resp,
					},
				},
			},
		},
	})

	if err := store.SaveSession(ctx, sess2); err != nil {
		t.Fatalf("SaveSession (update with func response) failed: %v", err)
	}

	// 5. Verify order and content
	sess3, err := store.GetSession(ctx, "test-session")
	if err != nil {
		t.Fatalf("GetSession (final) failed: %v", err)
	}
	if len(sess3.Messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(sess3.Messages))
	}
	if sess3.Messages[1].Parts[0].GetFunctionCall().Name != "my-func" {
		t.Errorf("Expected second message function call 'my-func', got '%s'", sess3.Messages[1].Parts[0].GetFunctionCall().GetName())
	}
	if sess3.Messages[2].Parts[0].GetFunctionResponse().Response.GetFields()["result"].GetStringValue() != "ok" {
		t.Errorf("Expected third message function response 'ok', got '%v'", sess3.Messages[2].Parts[0].GetFunctionResponse().GetResponse())
	}

	// 6. List sessions
	sessions, err := store.ListSessions(ctx)
	if err != nil {
		t.Fatalf("ListSessions failed: %v", err)
	}
	if len(sessions) != 1 {
		t.Errorf("Expected 1 session in list, got %d", len(sessions))
	}
}
