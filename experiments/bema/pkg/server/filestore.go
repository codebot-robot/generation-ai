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
	"google.golang.org/protobuf/encoding/protojson"
)

// FileSessionStore is an implementation of SessionStore that uses the filesystem.
type FileSessionStore struct {
	storageDir string
	mu         sync.RWMutex
}

// NewFileSessionStore creates a new FileSessionStore.
func NewFileSessionStore(storageDir string) (*FileSessionStore, error) {
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		return nil, err
	}
	return &FileSessionStore{storageDir: storageDir}, nil
}

// GetSession retrieves a session by ID.
func (s *FileSessionStore) GetSession(ctx context.Context, id string) (*pb.Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path := filepath.Join(s.storageDir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	session := &pb.Session{}
	if err := protojson.Unmarshal(data, session); err != nil {
		return nil, err
	}
	return session, nil
}

// SaveSession updates an existing session.
func (s *FileSessionStore) SaveSession(ctx context.Context, session *pb.Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	m := protojson.MarshalOptions{Multiline: true}
	data, err := m.Marshal(session)
	if err != nil {
		return err
	}

	path := filepath.Join(s.storageDir, session.Id+".json")
	return os.WriteFile(path, data, 0644)
}

// ListSessions returns a list of sessions.
func (s *FileSessionStore) ListSessions(ctx context.Context) ([]*pb.Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := os.ReadDir(s.storageDir)
	if err != nil {
		return nil, err
	}

	var sessions []*pb.Session
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		path := filepath.Join(s.storageDir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		session := &pb.Session{}
		if err := protojson.Unmarshal(data, session); err != nil {
			continue
		}
		sessions = append(sessions, session)
	}
	return sessions, nil
}
