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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	bemav1alpha1 "github.com/gke-labs/generation-ai/experiments/bema/pkg/apis/v1alpha1"
	"google.golang.org/protobuf/types/known/timestamppb"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type mockStore struct {
	sessions []*pb.Session
}

func (m *mockStore) GetSession(ctx context.Context, id string) (*pb.Session, error) {
	for _, s := range m.sessions {
		if s.Id == id {
			return s, nil
		}
	}
	return nil, nil
}

func (m *mockStore) SaveSession(ctx context.Context, session *pb.Session) error {
	m.sessions = append(m.sessions, session)
	return nil
}

func (m *mockStore) ListSessions(ctx context.Context) ([]*pb.Session, error) {
	return m.sessions, nil
}

func TestAPIServer_Discovery(t *testing.T) {
	s := NewAPIServer(&mockStore{})

	tests := []struct {
		path string
		kind string
	}{
		{"/apis", "APIGroupList"},
		{"/apis/bema.labs.gke.io", "APIGroup"},
		{"/apis/bema.labs.gke.io/v1alpha1", "APIResourceList"},
	}

	for _, tt := range tests {
		req := httptest.NewRequest("GET", tt.path, nil)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("%s: expected status 200, got %d", tt.path, w.Code)
		}

		var resp map[string]interface{}
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("%s: failed to unmarshal: %v", tt.path, err)
		}
		if resp["kind"] != tt.kind {
			t.Errorf("%s: expected kind %s, got %s", tt.path, tt.kind, resp["kind"])
		}
	}
}

func TestAPIServer_Resources(t *testing.T) {
	now := timestamppb.Now()
	store := &mockStore{
		sessions: []*pb.Session{
			{
				Id:        "test-session",
				Status:    "active",
				CreatedAt: now,
				Messages: []*pb.Message{
					{
						Role:      "user",
						Timestamp: now,
						Parts: []*pb.Part{
							{Data: &pb.Part_Text{Text: "hello"}},
						},
					},
				},
			},
		},
	}
	s := NewAPIServer(store)

	t.Run("List ChatSessions", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/apis/bema.labs.gke.io/v1alpha1/namespaces/default/chatsessions", nil)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var list bemav1alpha1.ChatSessionList
		if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		if len(list.Items) != 1 {
			t.Errorf("expected 1 session, got %d", len(list.Items))
		}
		if list.Items[0].Name != "test-session" {
			t.Errorf("expected name test-session, got %s", list.Items[0].Name)
		}
	})

	t.Run("Get ChatSession", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/apis/bema.labs.gke.io/v1alpha1/namespaces/default/chatsessions/test-session", nil)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var cs bemav1alpha1.ChatSession
		if err := json.Unmarshal(w.Body.Bytes(), &cs); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		if cs.Name != "test-session" {
			t.Errorf("expected name test-session, got %s", cs.Name)
		}
	})

	t.Run("List ChatSessionMessages", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/apis/bema.labs.gke.io/v1alpha1/namespaces/default/chatsessionmessages", nil)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var list bemav1alpha1.ChatSessionMessageList
		if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		if len(list.Items) != 1 {
			t.Errorf("expected 1 message, got %d", len(list.Items))
		}
		if list.Items[0].Name != "test-session-0" {
			t.Errorf("expected name test-session-0, got %s", list.Items[0].Name)
		}
	})

	t.Run("Get ChatSessionMessage", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/apis/bema.labs.gke.io/v1alpha1/namespaces/default/chatsessionmessages/test-session-0", nil)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var csm bemav1alpha1.ChatSessionMessage
		if err := json.Unmarshal(w.Body.Bytes(), &csm); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		if csm.Name != "test-session-0" {
			t.Errorf("expected name test-session-0, got %s", csm.Name)
		}
		if csm.Spec.SessionID != "test-session" {
			t.Errorf("expected session ID test-session, got %s", csm.Spec.SessionID)
		}
	})

	t.Run("Table support", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/apis/bema.labs.gke.io/v1alpha1/namespaces/default/chatsessions", nil)
		req.Header.Set("Accept", "application/json;as=Table;v=v1;g=meta.k8s.io")
		w := httptest.NewRecorder()
		s.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var table metav1.Table
		if err := json.Unmarshal(w.Body.Bytes(), &table); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		if table.Kind != "Table" {
			t.Errorf("expected kind Table, got %s", table.Kind)
		}
		if len(table.ColumnDefinitions) != 3 {
			t.Errorf("expected 3 columns, got %d", len(table.ColumnDefinitions))
		}
		if len(table.Rows) != 1 {
			t.Errorf("expected 1 row, got %d", len(table.Rows))
		}
	})
}
