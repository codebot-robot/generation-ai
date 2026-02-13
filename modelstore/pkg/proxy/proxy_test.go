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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
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

	p, err := NewProxy(upstream.URL, cacheDir)
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
