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
		if !messageList.Items[i].Spec.Timestamp.Equal(&messageList.Items[j].Spec.Timestamp) {
			return messageList.Items[i].Spec.Timestamp.Before(&messageList.Items[j].Spec.Timestamp)
		}
		return messageList.Items[i].Name < messageList.Items[j].Name
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

		for _, part := range msg.Spec.Parts {
			pbPart := &pb.Part{
				Thought:          part.Thought,
				ThoughtSignature: part.ThoughtSignature,
			}
			if part.Text != "" {
				pbPart.Data = &pb.Part_Text{Text: part.Text}
			} else if part.FunctionRequest != nil {
				args, err := rawExtensionToStruct(part.FunctionRequest.Args)
				if err != nil {
					return nil, fmt.Errorf("converting function request args: %w", err)
				}
				pbPart.Data = &pb.Part_FunctionCall{
					FunctionCall: &pb.FunctionCall{
						Name: part.FunctionRequest.Name,
						Args: args,
					},
				}
			} else if part.FunctionResponse != nil {
				response, err := rawExtensionToStruct(part.FunctionResponse.Response)
				if err != nil {
					return nil, fmt.Errorf("converting function response: %w", err)
				}
				pbPart.Data = &pb.Part_FunctionResponse{
					FunctionResponse: &pb.FunctionResponse{
						Name:     part.FunctionResponse.Name,
						Response: response,
					},
				}
			}
			pbMsg.Parts = append(pbMsg.Parts, pbPart)
		}
		session.Messages = append(session.Messages, pbMsg)
	}

	return session, nil
}

func structToRawExtension(s *structpb.Struct) (runtime.RawExtension, error) {
	if s == nil {
		return runtime.RawExtension{}, nil
	}
	b, err := protojson.Marshal(s)
	if err != nil {
		return runtime.RawExtension{}, err
	}
	return runtime.RawExtension{Raw: b}, nil
}

func rawExtensionToStruct(re runtime.RawExtension) (*structpb.Struct, error) {
	if re.Raw == nil {
		return nil, nil
	}
	s := &structpb.Struct{}
	if err := protojson.Unmarshal(re.Raw, s); err != nil {
		return nil, err
	}
	return s, nil
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
			},
		}
		if msg.Timestamp != nil {
			chatMsg.Spec.Timestamp = metav1.NewTime(msg.Timestamp.AsTime())
		}
		for _, part := range msg.Parts {
			p := bemav1alpha1.Part{
				Thought:          part.Thought,
				ThoughtSignature: part.ThoughtSignature,
			}
			switch d := part.Data.(type) {
			case *pb.Part_Text:
				p.Text = d.Text
			case *pb.Part_FunctionCall:
				args, err := structToRawExtension(d.FunctionCall.Args)
				if err != nil {
					return fmt.Errorf("converting function call args: %w", err)
				}
				p.FunctionRequest = &bemav1alpha1.FunctionCall{
					Name: d.FunctionCall.Name,
					Args: args,
				}
			case *pb.Part_FunctionResponse:
				response, err := structToRawExtension(d.FunctionResponse.Response)
				if err != nil {
					return fmt.Errorf("converting function response: %w", err)
				}
				p.FunctionResponse = &bemav1alpha1.FunctionResponse{
					Name:     d.FunctionResponse.Name,
					Response: response,
				}
			}
			chatMsg.Spec.Parts = append(chatMsg.Spec.Parts, p)
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
