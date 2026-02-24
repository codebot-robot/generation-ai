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

package server

import (
	"context"
	"os"
	"testing"
	"time"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestBemaServer(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bema-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	s, err := NewBemaServer(tmpDir, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()

	// 1. Create session
	sess, err := s.CreateSession(ctx, &pb.CreateSessionRequest{})
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if sess.Id == "" {
		t.Error("Expected non-empty session ID")
	}

	// 2. Append message
	_, err = s.AppendMessage(ctx, &pb.AppendMessageRequest{
		Id: sess.Id,
		Message: &pb.Message{
			Role: "user",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_Text{
						Text: "Hello",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("AppendMessage failed: %v", err)
	}

	// 3. Get session and verify
	sess2, err := s.GetSession(ctx, &pb.GetSessionRequest{Id: sess.Id})
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if len(sess2.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(sess2.Messages))
	}
	if sess2.Messages[0].Parts[0].GetText() != "Hello" {
		t.Errorf("Expected message content 'Hello', got '%s'", sess2.Messages[0].Parts[0].GetText())
	}

	// 4. Persistence test
	s2, err := NewBemaServer(tmpDir, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	sess3, err := s2.GetSession(ctx, &pb.GetSessionRequest{Id: sess.Id})
	if err != nil {
		t.Fatalf("GetSession after reload failed: %v", err)
	}
	if len(sess3.Messages) != 1 {
		t.Errorf("Expected 1 message after reload, got %d", len(sess3.Messages))
	}
}

type mockWatchServer struct {
	pb.BemaService_WatchSessionServer
	ctx    context.Context
	events chan *pb.SessionEvent
}

func (m *mockWatchServer) Send(ev *pb.SessionEvent) error {
	m.events <- ev
	return nil
}

func (m *mockWatchServer) Context() context.Context {
	return m.ctx
}

func TestWatchSession(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bema-watch-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	s, err := NewBemaServer(tmpDir, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sess, _ := s.CreateSession(ctx, &pb.CreateSessionRequest{})

	events := make(chan *pb.SessionEvent, 10)
	mock := &mockWatchServer{ctx: ctx, events: events}

	go func() {
		if err := s.WatchSession(&pb.WatchSessionRequest{Id: sess.Id}, mock); err != nil {
			t.Logf("WatchSession exited: %v", err)
		}
	}()

	// Wait for initial UPDATED event
	select {
	case ev := <-events:
		if ev.Type != pb.SessionEvent_UPDATED {
			t.Errorf("Expected UPDATED event, got %v", ev.Type)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for initial event")
	}

	// Append a message and wait for event
	s.AppendMessage(ctx, &pb.AppendMessageRequest{
		Id: sess.Id,
		Message: &pb.Message{
			Role: "user",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_Text{
						Text: "Watch me",
					},
				},
			},
		},
	})

	select {
	case ev := <-events:
		if ev.Type != pb.SessionEvent_MESSAGE_APPENDED {
			t.Errorf("Expected MESSAGE_APPENDED event, got %v", ev.Type)
		}
		if len(ev.Session.Messages) != 1 {
			t.Errorf("Expected 1 message in event session, got %d", len(ev.Session.Messages))
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for message event")
	}
}

type mockBackend struct {
	triggered chan bool
	response  *pb.Message
}

func (m *mockBackend) GenerateResponse(ctx context.Context, session *pb.Session) (*pb.Message, error) {
	m.triggered <- true
	return m.response, nil
}

func TestBackendTriggered(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bema-backend-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	triggered := make(chan bool, 1)
	mock := &mockBackend{
		triggered: triggered,
		response: &pb.Message{
			Role: "model",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_Text{
						Text: "I am a robot",
					},
				},
			},
		},
	}

	s, err := NewBemaServer(tmpDir, mock, nil)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	sess, _ := s.CreateSession(ctx, &pb.CreateSessionRequest{})

	s.AppendMessage(ctx, &pb.AppendMessageRequest{
		Id: sess.Id,
		Message: &pb.Message{
			Role: "user",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_Text{
						Text: "Hello",
					},
				},
			},
		},
	})

	select {
	case <-triggered:
		// success
	case <-time.After(2 * time.Second):
		t.Fatal("Backend was not triggered")
	}

	// Verify the assistant message was appended
	time.Sleep(100 * time.Millisecond) // Give it a moment to append
	sess2, _ := s.GetSession(ctx, &pb.GetSessionRequest{Id: sess.Id})
	if len(sess2.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(sess2.Messages))
	}
	if sess2.Messages[1].Role != "model" {
		t.Errorf("Expected second message role 'model', got '%s'", sess2.Messages[1].Role)
	}
}

type mockExecutor struct {
	triggered chan bool
	response  *pb.Message
}

func (m *mockExecutor) Execute(ctx context.Context, sessionID string, message *pb.Message) (*pb.Message, error) {
	m.triggered <- true
	return m.response, nil
}

func TestToolCalling(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bema-tool-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	backendTriggered := make(chan bool, 2)
	executorTriggered := make(chan bool, 1)

	args, _ := structpb.NewStruct(map[string]any{
		"command": "ls",
	})
	backend := &mockBackend{
		triggered: backendTriggered,
		response: &pb.Message{
			Role: "model",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_FunctionCall{
						FunctionCall: &pb.FunctionCall{
							Name: "exec",
							Args: args,
						},
					},
				},
			},
		},
	}

	response, _ := structpb.NewStruct(map[string]any{
		"output": "file1.txt",
	})
	executor := &mockExecutor{
		triggered: executorTriggered,
		response: &pb.Message{
			Role: "function",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_FunctionResponse{
						FunctionResponse: &pb.FunctionResponse{
							Name:     "exec",
							Response: response,
						},
					},
				},
			},
		},
	}

	s, err := NewBemaServer(tmpDir, backend, executor)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	sess, _ := s.CreateSession(ctx, &pb.CreateSessionRequest{})

	// Change backend response for the second call
	go func() {
		<-backendTriggered
		backend.response = &pb.Message{
			Role: "model",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_Text{
						Text: "Done",
					},
				},
			},
		}
	}()

	s.AppendMessage(ctx, &pb.AppendMessageRequest{
		Id: sess.Id,
		Message: &pb.Message{
			Role: "user",
			Parts: []*pb.Part{
				{
					Data: &pb.Part_Text{
						Text: "Run ls",
					},
				},
			},
		},
	})

	select {
	case <-executorTriggered:
		// success
	case <-time.After(2 * time.Second):
		t.Fatal("Executor was not triggered")
	}

	select {
	case <-backendTriggered:
		// second call success
	case <-time.After(2 * time.Second):
		t.Fatal("Backend was not triggered second time")
	}

	// Verify messages
	time.Sleep(100 * time.Millisecond)
	sess2, _ := s.GetSession(ctx, &pb.GetSessionRequest{Id: sess.Id})
	// 1: user, 2: assistant (tool call), 3: tool (output), 4: assistant (final)
	if len(sess2.Messages) != 4 {
		t.Errorf("Expected 4 messages, got %d", len(sess2.Messages))
	}
}
