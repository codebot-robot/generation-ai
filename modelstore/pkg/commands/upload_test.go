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

package commands

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildUploadCommand(t *testing.T) {
	cmd := BuildUploadCommand()
	if cmd == nil {
		t.Fatal("expected non-nil command")
	}

	modelNameFlag := cmd.Flag("model-name")
	if modelNameFlag == nil {
		t.Fatal("missing flag model-name")
	}
	modelDirFlag := cmd.Flag("model-dir")
	if modelDirFlag == nil {
		t.Fatal("missing flag model-dir")
	}
}

func TestRunUpload(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "upload-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer func() {
		_ = os.RemoveAll(tmpDir)
	}()

	testContent := []byte("hello model weights")
	testFile := filepath.Join(tmpDir, "weights.bin")
	if err := os.WriteFile(testFile, testContent, 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	h := sha256.New()
	_, err = h.Write(testContent)
	if err != nil {
		t.Fatalf("failed to hash test content: %v", err)
	}
	expectedSha := hex.EncodeToString(h.Sum(nil))

	uploadedBlobs := make(map[string]bool)
	createdModels := make(map[string]bool)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/blobs/") {
			sha := strings.TrimPrefix(r.URL.Path, "/blobs/")
			if r.Method == http.MethodHead {
				if uploadedBlobs[sha] {
					w.WriteHeader(http.StatusOK)
				} else {
					w.WriteHeader(http.StatusNotFound)
				}
				return
			}
			if r.Method == http.MethodPut {
				uploadedBlobs[sha] = true
				w.WriteHeader(http.StatusCreated)
				return
			}
		}
		if r.URL.Path == "/models" && r.Method == http.MethodPost {
			createdModels["test-model"] = true
			w.WriteHeader(http.StatusCreated)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	opt := &UploadOptions{
		ModelstoreURL: server.URL,
		ModelName:     "test-model",
		ModelDir:      tmpDir,
	}

	ctx := context.Background()
	if err := RunUpload(ctx, opt); err != nil {
		t.Fatalf("RunUpload failed: %v", err)
	}

	if !uploadedBlobs[expectedSha] {
		t.Errorf("expected blob %s to be uploaded", expectedSha)
	}
	if !createdModels["test-model"] {
		t.Errorf("expected model test-model to be created")
	}
}

func TestRunUpload_UploadBlobFailure(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "upload-test-err-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer func() {
		_ = os.RemoveAll(tmpDir)
	}()

	testContent := []byte("error content")
	testFile := filepath.Join(tmpDir, "weights.bin")
	if err := os.WriteFile(testFile, testContent, 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/blobs/") && r.Method == http.MethodPut {
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	opt := &UploadOptions{
		ModelstoreURL: server.URL,
		ModelName:     "test-model",
		ModelDir:      tmpDir,
	}

	err = RunUpload(context.Background(), opt)
	if err == nil {
		t.Fatal("expected error on failed blob upload, got nil")
	}
	if !strings.Contains(err.Error(), "failed to upload blob") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestRunUpload_CreateModelFailure(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "upload-test-err-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer func() {
		_ = os.RemoveAll(tmpDir)
	}()

	testContent := []byte("error content")
	testFile := filepath.Join(tmpDir, "weights.bin")
	if err := os.WriteFile(testFile, testContent, 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/blobs/") && r.Method == http.MethodPut {
			w.WriteHeader(http.StatusCreated)
			return
		}
		if r.URL.Path == "/models" && r.Method == http.MethodPost {
			http.Error(w, "conflict creating model", http.StatusConflict)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	opt := &UploadOptions{
		ModelstoreURL: server.URL,
		ModelName:     "test-model",
		ModelDir:      tmpDir,
	}

	err = RunUpload(context.Background(), opt)
	if err == nil {
		t.Fatal("expected error on failed model creation, got nil")
	}
	if !strings.Contains(err.Error(), "failed to create model") {
		t.Errorf("unexpected error message: %v", err)
	}
}
