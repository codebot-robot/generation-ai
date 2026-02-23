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
	"path/filepath"
	"sync"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/timestamppb"
	"k8s.io/klog/v2"
)

type BemaServer struct {
	pb.UnimplementedBemaServiceServer

	storageDir string
	backend    Backend

	mu       sync.RWMutex
	sessions map[string]*pb.Session
	watchers map[string][]chan *pb.SessionEvent
}

func NewBemaServer(storageDir string, backend Backend) (*BemaServer, error) {
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		return nil, err
	}

	s := &BemaServer{
		storageDir: storageDir,
		backend:    backend,
		sessions:   make(map[string]*pb.Session),
		watchers:   make(map[string][]chan *pb.SessionEvent),
	}

	if err := s.loadSessions(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *BemaServer) loadSessions() error {
	entries, err := os.ReadDir(s.storageDir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		path := filepath.Join(s.storageDir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			klog.ErrorS(err, "failed to read session file", "path", path)
			continue
		}

		session := &pb.Session{}
		if err := protojson.Unmarshal(data, session); err != nil {
			klog.ErrorS(err, "failed to unmarshal session", "path", path)
			continue
		}

		s.sessions[session.Id] = session
	}
	return nil
}

func (s *BemaServer) saveSession(session *pb.Session) error {
	m := protojson.MarshalOptions{Multiline: true}
	data, err := m.Marshal(session)
	if err != nil {
		return err
	}

	path := filepath.Join(s.storageDir, session.Id+".json")
	return os.WriteFile(path, data, 0644)
}

func (s *BemaServer) CreateSession(ctx context.Context, req *pb.CreateSessionRequest) (*pb.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := uuid.New().String()
	now := timestamppb.Now()

	session := &pb.Session{
		Id:        id,
		CreatedAt: now,
		UpdatedAt: now,
		Config:    req.Config,
		Status:    "idle",
	}

	if err := s.saveSession(session); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to save session: %v", err)
	}

	s.sessions[id] = session
	return session, nil
}

func (s *BemaServer) GetSession(ctx context.Context, req *pb.GetSessionRequest) (*pb.Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	session, ok := s.sessions[req.Id]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "session %s not found", req.Id)
	}

	return session, nil
}

func (s *BemaServer) ListSessions(ctx context.Context, req *pb.ListSessionsRequest) (*pb.ListSessionsResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sessions := make([]*pb.Session, 0, len(s.sessions))
	for _, session := range s.sessions {
		sessions = append(sessions, session)
	}

	return &pb.ListSessionsResponse{Sessions: sessions}, nil
}

func (s *BemaServer) AppendMessage(ctx context.Context, req *pb.AppendMessageRequest) (*pb.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	session, ok := s.sessions[req.Id]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "session %s not found", req.Id)
	}

	msg := req.Message
	if msg.Timestamp == nil {
		msg.Timestamp = timestamppb.Now()
	}

	session.Messages = append(session.Messages, msg)
	session.UpdatedAt = timestamppb.Now()

	if err := s.saveSession(session); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to save session: %v", err)
	}

	s.notifyWatchers(session, pb.SessionEvent_MESSAGE_APPENDED)

	if s.backend != nil && msg.Role == "user" {
		go s.generateBackendResponse(context.Background(), session.Id)
	}

	return session, nil
}

func (s *BemaServer) generateBackendResponse(ctx context.Context, sessionID string) {
	s.mu.RLock()
	session, ok := s.sessions[sessionID]
	if !ok {
		s.mu.RUnlock()
		return
	}
	s.mu.RUnlock()

	resp, err := s.backend.GenerateResponse(ctx, session)
	if err != nil {
		klog.ErrorS(err, "failed to generate backend response", "sessionID", sessionID)
		return
	}

	// Append the response
	_, err = s.AppendMessage(ctx, &pb.AppendMessageRequest{
		Id:      sessionID,
		Message: resp,
	})
	if err != nil {
		klog.ErrorS(err, "failed to append backend response", "sessionID", sessionID)
	}
}

func (s *BemaServer) WatchSession(req *pb.WatchSessionRequest, stream pb.BemaService_WatchSessionServer) error {
	s.mu.Lock()
	session, ok := s.sessions[req.Id]
	if !ok {
		s.mu.Unlock()
		return status.Errorf(codes.NotFound, "session %s not found", req.Id)
	}

	ch := make(chan *pb.SessionEvent, 10)
	s.watchers[req.Id] = append(s.watchers[req.Id], ch)
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		watchers := s.watchers[req.Id]
		for i, w := range watchers {
			if w == ch {
				s.watchers[req.Id] = append(watchers[:i], watchers[i+1:]...)
				break
			}
		}
		close(ch)
	}()

	// Send initial state
	if err := stream.Send(&pb.SessionEvent{
		Type:    pb.SessionEvent_UPDATED,
		Session: session,
	}); err != nil {
		return err
	}

	for {
		select {
		case event := <-ch:
			if err := stream.Send(event); err != nil {
				return err
			}
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
	}
}

func (s *BemaServer) notifyWatchers(session *pb.Session, eventType pb.SessionEvent_Type) {
	event := &pb.SessionEvent{
		Type:    eventType,
		Session: session,
	}

	for _, ch := range s.watchers[session.Id] {
		select {
		case ch <- event:
		default:
			klog.V(2).InfoS("watcher channel full, dropping event", "sessionId", session.Id)
		}
	}
}
