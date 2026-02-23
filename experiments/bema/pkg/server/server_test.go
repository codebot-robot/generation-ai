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
)

func TestBemaServer(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bema-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	s, err := NewBemaServer(tmpDir)
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
			Role:    "user",
			Content: "Hello",
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
	if sess2.Messages[0].Content != "Hello" {
		t.Errorf("Expected message content 'Hello', got '%s'", sess2.Messages[0].Content)
	}

	// 4. Persistence test
	s2, err := NewBemaServer(tmpDir)
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

	s, err := NewBemaServer(tmpDir)
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
			Role:    "user",
			Content: "Watch me",
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
