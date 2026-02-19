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

package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestProxy(t *testing.T) {
	// Create a temporary cache directory
	cacheDir, err := os.MkdirTemp("", "modelstore-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(cacheDir)

	// Create a mock upstream server
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/test-file" {
			fmt.Fprint(w, "upstream content")
			return
		}
		http.NotFound(w, r)
	}))
	defer upstream.Close()

	p, err := NewProxy(upstream.URL, cacheDir, nil)
	if err != nil {
		t.Fatalf("Failed to create proxy: %v", err)
	}

	// First request - should fetch from upstream and cache
	req := httptest.NewRequest(http.MethodGet, "/test-file", nil)
	w := httptest.NewRecorder()
	p.ServeHTTP(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)
	if string(body) != "upstream content" {
		t.Errorf("Expected 'upstream content', got '%s'", string(body))
	}

	// Verify it's in cache
	cachePath := filepath.Join(cacheDir, "test-file")
	if _, err := os.Stat(cachePath); os.IsNotExist(err) {
		t.Errorf("Expected file to be cached at %s", cachePath)
	}

	// Second request - should serve from cache
	// Change upstream content to verify it's NOT used
	// (Actually easier to just shut down upstream or check logs)
	upstream.Close() // Upstream is now closed, if it tries to fetch it will fail

	w2 := httptest.NewRecorder()
	p.ServeHTTP(w2, req)

	resp2 := w2.Result()
	body2, _ := io.ReadAll(resp2.Body)
	if string(body2) != "upstream content" {
		t.Errorf("Expected 'upstream content' from cache, got '%s'", string(body2))
	}
}

func TestProxyPut(t *testing.T) {
	// Create a temporary cache directory
	cacheDir, err := os.MkdirTemp("", "modelstore-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(cacheDir)

	p, err := NewProxy("http://localhost:1234", cacheDir, nil) // Upstream doesn't matter for PUT
	if err != nil {
		t.Fatalf("Failed to create proxy: %v", err)
	}

	content := "finetuned model content"
	req := httptest.NewRequest(http.MethodPut, "/finetuned/model.bin", strings.NewReader(content))
	w := httptest.NewRecorder()
	p.ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusCreated {
		t.Errorf("Expected status 201 Created, got %d", resp.StatusCode)
	}

	// Verify it's in cache
	cachePath := filepath.Join(cacheDir, "finetuned/model.bin")
	gotContent, err := os.ReadFile(cachePath)
	if err != nil {
		t.Fatalf("Failed to read cached file: %v", err)
	}
	if string(gotContent) != content {
		t.Errorf("Expected '%s', got '%s'", content, string(gotContent))
	}
}

func TestBlobAndModelAPI(t *testing.T) {
	cacheDir, err := os.MkdirTemp("", "modelstore-blob-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(cacheDir)

	scheme := runtime.NewScheme()
	_ = v1alpha1.AddToScheme(scheme)
	kube := fake.NewClientBuilder().WithScheme(scheme).Build()

	p, err := NewProxy("http://localhost:1234", cacheDir, kube)
	if err != nil {
		t.Fatalf("Failed to create proxy: %v", err)
	}

	// 1. Upload a blob
	content := "blob content"
	sha := "7b24cf3d897fd680e0258c1c7c23db50a5428581ed1785c08de505c381b4c4b5"
	req := httptest.NewRequest(http.MethodPut, "/blobs/"+sha, strings.NewReader(content))
	w := httptest.NewRecorder()
	p.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("Expected blob PUT status 201, got %d", w.Code)
	}

	// 2. Create a model
	model := v1alpha1.Model{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-model",
		},
		Spec: v1alpha1.ModelSpec{
			Files: []v1alpha1.File{
				{Path: "weights.bin", SHA256: sha},
			},
		},
	}
	modelJSON, _ := json.Marshal(model)
	req = httptest.NewRequest(http.MethodPost, "/models", bytes.NewReader(modelJSON))
	w = httptest.NewRecorder()
	p.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("Expected model POST status 201, got %d: %s", w.Code, w.Body.String())
	}

	// 3. List models
	req = httptest.NewRequest(http.MethodGet, "/models", nil)
	w = httptest.NewRecorder()
	p.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected model GET status 200, got %d", w.Code)
	}

	var list v1alpha1.ModelList
	if err := json.NewDecoder(w.Body).Decode(&list); err != nil {
		t.Fatalf("Failed to decode model list: %v", err)
	}

	if len(list.Items) != 1 {
		t.Errorf("Expected 1 model, got %d", len(list.Items))
	}
	if list.Items[0].Name != "test-model" {
		t.Errorf("Expected model name 'test-model', got '%s'", list.Items[0].Name)
	}
}
