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
	"context"
	"encoding/json"
	"fmt"
	"sort"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	bemav1alpha1 "github.com/gke-labs/generation-ai/experiments/bema/pkg/apis/v1alpha1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const (
	// LabelSessionID is the label used to filter messages by session ID.
	LabelSessionID = "bema.labs.gke.io/session-id"
)

// K8sSessionStore is an implementation of SessionStore that uses Kubernetes CRDs.
type K8sSessionStore struct {
	client    client.Client
	namespace string
}

// NewK8sSessionStore creates a new K8sSessionStore.
func NewK8sSessionStore(c client.Client, namespace string) *K8sSessionStore {
	return &K8sSessionStore{
		client:    c,
		namespace: namespace,
	}
}

// GetSession retrieves a session by ID.
func (s *K8sSessionStore) GetSession(ctx context.Context, id string) (*pb.Session, error) {
	chatSession := &bemav1alpha1.ChatSession{}
	if err := s.client.Get(ctx, types.NamespacedName{Name: id, Namespace: s.namespace}, chatSession); err != nil {
		if client.IgnoreNotFound(err) == nil {
			return nil, nil
		}
		return nil, fmt.Errorf("getting ChatSession: %w", err)
	}

	messageList := &bemav1alpha1.ChatSessionMessageList{}
	if err := s.client.List(ctx, messageList, client.InNamespace(s.namespace), client.MatchingLabels{LabelSessionID: id}); err != nil {
		return nil, fmt.Errorf("listing ChatSessionMessages: %w", err)
	}

	sort.Slice(messageList.Items, func(i, j int) bool {
		return messageList.Items[i].Spec.Index < messageList.Items[j].Spec.Index
	})

	session := &pb.Session{
		Id:        chatSession.Name,
		Status:    chatSession.Status.Status,
		CreatedAt: timestamppb.New(chatSession.CreationTimestamp.Time),
		UpdatedAt: timestamppb.New(chatSession.CreationTimestamp.Time), // Fallback
	}

	if chatSession.Spec.Config.Raw != nil {
		config := &structpb.Struct{}
		if err := protojson.Unmarshal(chatSession.Spec.Config.Raw, config); err != nil {
			return nil, fmt.Errorf("unmarshaling config: %w", err)
		}
		session.Config = config
	}

	for _, msg := range messageList.Items {
		pbMsg := &pb.Message{
			Role: msg.Spec.Role,
		}
		if !msg.Spec.Timestamp.IsZero() {
			pbMsg.Timestamp = timestamppb.New(msg.Spec.Timestamp.Time)
		}

		if msg.Spec.Parts.Raw != nil {
			// Parts is stored as a JSON array of Part objects.
			// Reconstruct a JSON that matches pb.Message structure.
			partsJSON := fmt.Sprintf(`{"parts": %s}`, string(msg.Spec.Parts.Raw))
			if err := protojson.Unmarshal([]byte(partsJSON), pbMsg); err != nil {
				return nil, fmt.Errorf("unmarshaling parts for message %d: %w", msg.Spec.Index, err)
			}
		}
		session.Messages = append(session.Messages, pbMsg)
	}

	return session, nil
}

// SaveSession updates an existing session.
func (s *K8sSessionStore) SaveSession(ctx context.Context, session *pb.Session) error {
	chatSession := &bemav1alpha1.ChatSession{
		ObjectMeta: metav1.ObjectMeta{
			Name:      session.Id,
			Namespace: s.namespace,
		},
	}

	if session.Config != nil {
		configBytes, err := protojson.Marshal(session.Config)
		if err != nil {
			return fmt.Errorf("marshaling config: %w", err)
		}
		chatSession.Spec.Config = runtime.RawExtension{Raw: configBytes}
	}
	chatSession.Status.Status = session.Status

	existing := &bemav1alpha1.ChatSession{}
	err := s.client.Get(ctx, types.NamespacedName{Name: session.Id, Namespace: s.namespace}, existing)
	if err != nil {
		if client.IgnoreNotFound(err) != nil {
			return fmt.Errorf("checking for existing ChatSession: %w", err)
		}
		if err := s.client.Create(ctx, chatSession); err != nil {
			return fmt.Errorf("creating ChatSession: %w", err)
		}
	} else {
		chatSession.ResourceVersion = existing.ResourceVersion
		if err := s.client.Update(ctx, chatSession); err != nil {
			return fmt.Errorf("updating ChatSession: %w", err)
		}
	}

	// Save messages
	for i, msg := range session.Messages {
		msgName := fmt.Sprintf("%s-%d", session.Id, i)
		chatMsg := &bemav1alpha1.ChatSessionMessage{
			ObjectMeta: metav1.ObjectMeta{
				Name:      msgName,
				Namespace: s.namespace,
				Labels: map[string]string{
					LabelSessionID: session.Id,
				},
			},
			Spec: bemav1alpha1.ChatSessionMessageSpec{
				SessionID: session.Id,
				Role:      msg.Role,
				Index:     int32(i),
			},
		}
		if msg.Timestamp != nil {
			chatMsg.Spec.Timestamp = metav1.NewTime(msg.Timestamp.AsTime())
		}
		if len(msg.Parts) > 0 {
			// Use a dummy message to marshal only parts using protojson.
			dummy := &pb.Message{Parts: msg.Parts}
			msgBytes, err := protojson.Marshal(dummy)
			if err != nil {
				return fmt.Errorf("marshaling message %d: %w", i, err)
			}

			// Extract only the "parts" array.
			var data map[string]json.RawMessage
			if err := json.Unmarshal(msgBytes, &data); err != nil {
				return fmt.Errorf("unmarshaling message JSON into map: %w", err)
			}
			chatMsg.Spec.Parts = runtime.RawExtension{Raw: data["parts"]}
		}

		existingMsg := &bemav1alpha1.ChatSessionMessage{}
		err := s.client.Get(ctx, types.NamespacedName{Name: msgName, Namespace: s.namespace}, existingMsg)
		if err != nil {
			if client.IgnoreNotFound(err) != nil {
				return fmt.Errorf("checking for existing ChatSessionMessage: %w", err)
			}
			if err := s.client.Create(ctx, chatMsg); err != nil {
				return fmt.Errorf("creating ChatSessionMessage %d: %w", i, err)
			}
		} else {
			chatMsg.ResourceVersion = existingMsg.ResourceVersion
			if err := s.client.Update(ctx, chatMsg); err != nil {
				return fmt.Errorf("updating ChatSessionMessage %d: %w", i, err)
			}
		}
	}

	return nil
}

// ListSessions returns a list of sessions.
func (s *K8sSessionStore) ListSessions(ctx context.Context) ([]*pb.Session, error) {
	sessionList := &bemav1alpha1.ChatSessionList{}
	if err := s.client.List(ctx, sessionList, client.InNamespace(s.namespace)); err != nil {
		return nil, fmt.Errorf("listing ChatSessions: %w", err)
	}

	var sessions []*pb.Session
	for _, chatSession := range sessionList.Items {
		session, err := s.GetSession(ctx, chatSession.Name)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}
	return sessions, nil
}
