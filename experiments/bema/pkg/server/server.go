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
	"sync"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type BemaServer struct {
	pb.UnimplementedBemaServiceServer

	store    SessionStore
	backend  Backend
	executor Executor
	client   client.Client

	mu       sync.RWMutex
	watchers map[string][]chan *pb.SessionEvent
}

func NewBemaServer(store SessionStore, backend Backend, executor Executor, client client.Client) (*BemaServer, error) {
	s := &BemaServer{
		store:    store,
		backend:  backend,
		executor: executor,
		client:   client,
		watchers: make(map[string][]chan *pb.SessionEvent),
	}

	return s, nil
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

	if err := s.store.SaveSession(ctx, session); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to save session: %v", err)
	}

	return session, nil
}

func (s *BemaServer) GetSession(ctx context.Context, req *pb.GetSessionRequest) (*pb.Session, error) {
	session, err := s.store.GetSession(ctx, req.Id)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get session: %v", err)
	}
	if session == nil {
		return nil, status.Errorf(codes.NotFound, "session %s not found", req.Id)
	}

	return session, nil
}

func (s *BemaServer) ListSessions(ctx context.Context, req *pb.ListSessionsRequest) (*pb.ListSessionsResponse, error) {
	sessions, err := s.store.ListSessions(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list sessions: %v", err)
	}

	return &pb.ListSessionsResponse{Sessions: sessions}, nil
}

func (s *BemaServer) AppendMessage(ctx context.Context, req *pb.AppendMessageRequest) (*pb.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	session, err := s.store.GetSession(ctx, req.Id)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get session: %v", err)
	}
	if session == nil {
		return nil, status.Errorf(codes.NotFound, "session %s not found", req.Id)
	}

	msg := req.Message
	if msg.Timestamp == nil {
		msg.Timestamp = timestamppb.Now()
	}

	session.Messages = append(session.Messages, msg)
	session.UpdatedAt = timestamppb.Now()

	if err := s.store.SaveSession(ctx, session); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to save session: %v", err)
	}

	s.notifyWatchers(session, pb.SessionEvent_MESSAGE_APPENDED)

	if s.backend != nil && msg.Role == "user" {
		go s.generateBackendResponse(context.Background(), session.Id)
	}

	return session, nil
}

func (s *BemaServer) generateBackendResponse(ctx context.Context, sessionID string) {
	for {
		session, err := s.store.GetSession(ctx, sessionID)
		if err != nil {
			klog.ErrorS(err, "failed to get session", "sessionID", sessionID)
			return
		}
		if session == nil {
			klog.V(2).InfoS("session not found, stopping backend response generation", "sessionID", sessionID)
			return
		}

		resp, err := s.backend.GenerateResponse(ctx, session)
		if err != nil {
			klog.ErrorS(err, "failed to generate backend response", "sessionID", sessionID)
			return
		}

		// Append the response
		session, err = s.AppendMessage(ctx, &pb.AppendMessageRequest{
			Id:      sessionID,
			Message: resp,
		})
		if err != nil {
			klog.ErrorS(err, "failed to append backend response", "sessionID", sessionID)
			return
		}

		// If it's a tool call, execute it and loop
		hasFunctionCalls := false
		for _, p := range resp.Parts {
			if _, ok := p.Data.(*pb.Part_FunctionCall); ok {
				hasFunctionCalls = true
				break
			}
		}

		if hasFunctionCalls && s.executor != nil {
			actions, toolResp, err := s.executor.Execute(ctx, sessionID, resp)
			if err != nil {
				klog.ErrorS(err, "failed to execute tools", "sessionID", sessionID)
				return
			}

			// Save actions to K8s
			for _, action := range actions {
				action.Spec.Timestamp = metav1.NewTime(resp.Timestamp.AsTime())
				if action.Namespace == "" {
					ns := os.Getenv("BEMA_NAMESPACE")
					if ns == "" {
						ns = "bema"
					}
					action.Namespace = ns
				}
				if err := s.client.Create(ctx, action); err != nil {
					klog.ErrorS(err, "failed to create AgentAction", "actionName", action.Name)
				}
			}

			if toolResp != nil {
				_, err = s.AppendMessage(ctx, &pb.AppendMessageRequest{
					Id:      sessionID,
					Message: toolResp,
				})
				if err != nil {
					klog.ErrorS(err, "failed to append tool response", "sessionID", sessionID)
					return
				}
				// Continue loop to let LLM process tool results
				continue
			}
		}

		// If no more tool calls, we are done
		break
	}
}

func (s *BemaServer) WatchSession(req *pb.WatchSessionRequest, stream pb.BemaService_WatchSessionServer) error {
	session, err := s.store.GetSession(stream.Context(), req.Id)
	if err != nil {
		return status.Errorf(codes.Internal, "failed to get session: %v", err)
	}
	if session == nil {
		return status.Errorf(codes.NotFound, "session %s not found", req.Id)
	}

	s.mu.Lock()
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
