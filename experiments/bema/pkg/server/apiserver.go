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
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	bemav1alpha1 "github.com/gke-labs/generation-ai/experiments/bema/pkg/apis/v1alpha1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/timestamppb"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

// APIServer implements a simple Kubernetes-compatible REST API for Bema resources.
type APIServer struct {
	Store SessionStore
}

// NewAPIServer creates a new APIServer.
func NewAPIServer(store SessionStore) *APIServer {
	return &APIServer{Store: store}
}

func (s *APIServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	klog.V(4).Infof("APIServer request: %s %s", r.Method, r.URL.Path)

	path := r.URL.Path
	switch {
	case path == "/apis":
		s.handleAPIs(w)
	case path == "/apis/bema.labs.gke.io":
		s.handleBemaGroup(w)
	case path == "/apis/bema.labs.gke.io/v1alpha1":
		s.handleBemaV1alpha1(w)
	case strings.HasPrefix(path, "/apis/bema.labs.gke.io/v1alpha1/"):
		s.handleResource(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (s *APIServer) handleAPIs(w http.ResponseWriter) {
	sendJSON(w, &metav1.APIGroupList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIGroupList",
			APIVersion: "v1",
		},
		Groups: []metav1.APIGroup{
			{
				Name: "bema.labs.gke.io",
				Versions: []metav1.GroupVersionForDiscovery{
					{
						GroupVersion: "bema.labs.gke.io/v1alpha1",
						Version:      "v1alpha1",
					},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{
					GroupVersion: "bema.labs.gke.io/v1alpha1",
					Version:      "v1alpha1",
				},
			},
		},
	})
}

func (s *APIServer) handleBemaGroup(w http.ResponseWriter) {
	sendJSON(w, &metav1.APIGroup{
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIGroup",
			APIVersion: "v1",
		},
		Name: "bema.labs.gke.io",
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: "bema.labs.gke.io/v1alpha1",
				Version:      "v1alpha1",
			},
		},
		PreferredVersion: metav1.GroupVersionForDiscovery{
			GroupVersion: "bema.labs.gke.io/v1alpha1",
			Version:      "v1alpha1",
		},
	})
}

func (s *APIServer) handleBemaV1alpha1(w http.ResponseWriter) {
	sendJSON(w, &metav1.APIResourceList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIResourceList",
			APIVersion: "v1",
		},
		GroupVersion: "bema.labs.gke.io/v1alpha1",
		APIResources: []metav1.APIResource{
			{
				Name:         "chatsessions",
				SingularName: "chatsession",
				Namespaced:   true,
				Kind:         "ChatSession",
				Verbs:        []string{"get", "list"},
				ShortNames:   []string{"cs"},
			},
			{
				Name:         "chatsessionmessages",
				SingularName: "chatsessionmessage",
				Namespaced:   true,
				Kind:         "ChatSessionMessage",
				Verbs:        []string{"get", "list"},
				ShortNames:   []string{"csm"},
			},
		},
	})
}

func (s *APIServer) handleResource(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/apis/bema.labs.gke.io/v1alpha1/"), "/")

	var resource, ns, name string

	if parts[0] == "namespaces" {
		if len(parts) < 3 {
			http.NotFound(w, r)
			return
		}
		ns = parts[1]
		resource = parts[2]
		if len(parts) > 3 {
			name = parts[3]
		}
	} else {
		resource = parts[0]
		if len(parts) > 1 {
			name = parts[1]
		}
	}

	if name != "" {
		s.handleGet(w, r, resource, ns, name)
	} else {
		s.handleList(w, r, resource, ns)
	}
}

func (s *APIServer) handleList(w http.ResponseWriter, r *http.Request, resource, ns string) {
	sessions, err := s.Store.ListSessions(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	isTable := strings.Contains(r.Header.Get("Accept"), "as=Table")

	if isTable {
		s.handleTableList(w, r, resource, ns, sessions)
		return
	}

	switch resource {
	case "chatsessions":
		list := &bemav1alpha1.ChatSessionList{
			TypeMeta: metav1.TypeMeta{
				Kind:       "ChatSessionList",
				APIVersion: "bema.labs.gke.io/v1alpha1",
			},
			Items: []bemav1alpha1.ChatSession{},
		}
		for _, sess := range sessions {
			list.Items = append(list.Items, *convertToChatSession(sess, ns))
		}
		sendJSON(w, list)
	case "chatsessionmessages":
		list := &bemav1alpha1.ChatSessionMessageList{
			TypeMeta: metav1.TypeMeta{
				Kind:       "ChatSessionMessageList",
				APIVersion: "bema.labs.gke.io/v1alpha1",
			},
			Items: []bemav1alpha1.ChatSessionMessage{},
		}
		for _, sess := range sessions {
			for i, msg := range sess.Messages {
				list.Items = append(list.Items, *convertToChatSessionMessage(sess.Id, i, msg, ns))
			}
		}
		sendJSON(w, list)
	default:
		http.NotFound(w, r)
	}
}

func (s *APIServer) handleTableList(w http.ResponseWriter, r *http.Request, resource, ns string, sessions []*pb.Session) {
	table := &metav1.Table{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Table",
			APIVersion: "meta.k8s.io/v1",
		},
	}

	switch resource {
	case "chatsessions":
		table.ColumnDefinitions = []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string", Format: "name", Description: "Name of the session"},
			{Name: "Status", Type: "string", Description: "Status of the session"},
			{Name: "Age", Type: "string", Description: "Age of the session"},
		}
		for _, sess := range sessions {
			table.Rows = append(table.Rows, metav1.TableRow{
				Cells: []interface{}{sess.Id, sess.Status, translateTimestamp(sess.CreatedAt)},
				Object: runtime.RawExtension{
					Object: convertToChatSession(sess, ns),
				},
			})
		}
	case "chatsessionmessages":
		table.ColumnDefinitions = []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string", Format: "name", Description: "Name of the message"},
			{Name: "Session", Type: "string", Description: "Session ID"},
			{Name: "Role", Type: "string", Description: "Role of the message"},
			{Name: "Age", Type: "string", Description: "Age of the message"},
		}
		for _, sess := range sessions {
			for i, msg := range sess.Messages {
				table.Rows = append(table.Rows, metav1.TableRow{
					Cells: []interface{}{fmt.Sprintf("%s-%d", sess.Id, i), sess.Id, msg.Role, translateTimestamp(msg.Timestamp)},
					Object: runtime.RawExtension{
						Object: convertToChatSessionMessage(sess.Id, i, msg, ns),
					},
				})
			}
		}
	}
	sendJSON(w, table)
}

func translateTimestamp(ts *timestamppb.Timestamp) string {
	if ts == nil {
		return "<unknown>"
	}
	return time.Since(ts.AsTime()).Round(time.Second).String()
}

func (s *APIServer) handleGet(w http.ResponseWriter, r *http.Request, resource, ns, name string) {
	ctx := r.Context()
	switch resource {
	case "chatsessions":
		sess, err := s.Store.GetSession(ctx, name)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if sess == nil {
			http.NotFound(w, r)
			return
		}
		sendJSON(w, convertToChatSession(sess, ns))
	case "chatsessionmessages":
		lastDash := strings.LastIndex(name, "-")
		if lastDash == -1 {
			http.NotFound(w, r)
			return
		}
		sessionID := name[:lastDash]
		indexStr := name[lastDash+1:]
		var index int
		if _, err := fmt.Sscanf(indexStr, "%d", &index); err != nil {
			http.NotFound(w, r)
			return
		}

		sess, err := s.Store.GetSession(ctx, sessionID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if sess == nil || index < 0 || index >= len(sess.Messages) {
			http.NotFound(w, r)
			return
		}
		sendJSON(w, convertToChatSessionMessage(sess.Id, index, sess.Messages[index], ns))
	default:
		http.NotFound(w, r)
	}
}

func sendJSON(w http.ResponseWriter, obj interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(obj); err != nil {
		klog.Errorf("failed to encode JSON: %v", err)
	}
}

func convertToChatSession(sess *pb.Session, namespace string) *bemav1alpha1.ChatSession {
	if namespace == "" {
		namespace = "default"
	}
	cs := &bemav1alpha1.ChatSession{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ChatSession",
			APIVersion: "bema.labs.gke.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:              sess.Id,
			Namespace:         namespace,
			CreationTimestamp: metav1.NewTime(sess.CreatedAt.AsTime()),
		},
		Status: bemav1alpha1.ChatSessionStatus{
			Status: sess.Status,
		},
	}
	if sess.Config != nil {
		b, _ := protojson.Marshal(sess.Config)
		cs.Spec.Config = runtime.RawExtension{Raw: b}
	}
	return cs
}

func convertToChatSessionMessage(sessionID string, index int, msg *pb.Message, namespace string) *bemav1alpha1.ChatSessionMessage {
	if namespace == "" {
		namespace = "default"
	}
	name := fmt.Sprintf("%s-%d", sessionID, index)
	csm := &bemav1alpha1.ChatSessionMessage{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ChatSessionMessage",
			APIVersion: "bema.labs.gke.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         namespace,
			CreationTimestamp: metav1.NewTime(msg.Timestamp.AsTime()),
			Labels: map[string]string{
				"bema.labs.gke.io/session-id": sessionID,
			},
		},
		Spec: bemav1alpha1.ChatSessionMessageSpec{
			SessionID: sessionID,
			Role:      msg.Role,
		},
	}
	if msg.Timestamp != nil {
		csm.Spec.Timestamp = metav1.NewTime(msg.Timestamp.AsTime())
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
			b, _ := protojson.Marshal(d.FunctionCall.Args)
			p.FunctionRequest = &bemav1alpha1.FunctionCall{
				Name: d.FunctionCall.Name,
				Args: runtime.RawExtension{Raw: b},
			}
		case *pb.Part_FunctionResponse:
			b, _ := protojson.Marshal(d.FunctionResponse.Response)
			p.FunctionResponse = &bemav1alpha1.FunctionResponse{
				Name:     d.FunctionResponse.Name,
				Response: runtime.RawExtension{Raw: b},
			}
		}
		csm.Spec.Parts = append(csm.Spec.Parts, p)
	}
	return csm
}
