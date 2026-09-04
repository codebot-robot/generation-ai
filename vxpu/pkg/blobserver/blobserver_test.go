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

package blobserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func TestHandleBlob(t *testing.T) {
	tempDir := t.TempDir()
	s := &Server{
		cacheDir: tempDir,
	}

	// Create a fake blob file in the cache directory
	sha := "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" // sha256 of "hello"
	content := []byte("hello world from cached blob!")
	err := os.WriteFile(filepath.Join(tempDir, sha), content, 0644)
	if err != nil {
		t.Fatalf("failed to write test blob file: %v", err)
	}

	// Test regular HTTP GET request without Range header
	req, err := http.NewRequest("GET", "/blobs/"+sha, nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	rr := httptest.NewRecorder()
	s.handleBlob(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code 200, got %d", rr.Code)
	}
	if !bytes.Equal(rr.Body.Bytes(), content) {
		t.Errorf("Expected body %q, got %q", string(content), rr.Body.String())
	}

	// Test HTTP GET request with Range header (bytes=6-10)
	reqRange, err := http.NewRequest("GET", "/blobs/"+sha, nil)
	if err != nil {
		t.Fatalf("failed to create range request: %v", err)
	}
	reqRange.Header.Set("Range", "bytes=6-10")
	rrRange := httptest.NewRecorder()
	s.handleBlob(rrRange, reqRange)

	if rrRange.Code != http.StatusPartialContent {
		t.Errorf("Expected status code 206, got %d", rrRange.Code)
	}
	expectedSubStr := "world"
	if rrRange.Body.String() != expectedSubStr {
		t.Errorf("Expected range body %q, got %q", expectedSubStr, rrRange.Body.String())
	}
}

func TestCacheAndRewriteManifest(t *testing.T) {
	// Spin up a dummy HTTP server to serve the mock Hugging Face model weight files
	fileContent := []byte("fake model weight content")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(fileContent)
	}))
	defer ts.Close()

	tempDir := t.TempDir()
	s, err := NewServer(tempDir, 0)
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	defer s.Close()

	sha := "1ab0ee7e6a298b845e6e6dc4b2dd383a06ca030d42a9d7f251eba1306cb63a4f" // sha256 of "fake model weight content"
	manifestData := v1alpha1.Manifest{
		Format: "vxpu-manifest/v1alpha1",
		Source: v1alpha1.ManifestSource{
			Repo:      "test-repo",
			Revision:  "main",
			CommitSHA: "abcdef123456",
		},
		Config: map[string]any{
			"model_type": "llama",
		},
		Files: map[string]v1alpha1.ManifestFile{
			sha: {
				Size:   int64(len(fileContent)),
				Name:   "model.safetensors",
				Source: ts.URL + "/model.safetensors",
			},
		},
		Tensors: map[string]any{
			"layernorm": "some-tensor-metadata",
		},
	}

	manifestBytes, err := json.Marshal(manifestData)
	if err != nil {
		t.Fatalf("failed to marshal original manifest: %v", err)
	}

	routerIP := "127.0.0.1"
	rewrittenJSON, err := s.CacheAndRewriteManifest(t.Context(), string(manifestBytes), routerIP)
	if err != nil {
		t.Fatalf("CacheAndRewriteManifest failed: %v", err)
	}

	// Unmarshal back to check if all keys are completely preserved
	var result v1alpha1.Manifest
	if err := json.Unmarshal([]byte(rewrittenJSON), &result); err != nil {
		t.Fatalf("failed to unmarshal rewritten manifest: %v", err)
	}

	if result.Format != "vxpu-manifest/v1alpha1" {
		t.Errorf("Expected Format 'vxpu-manifest/v1alpha1', got %q", result.Format)
	}
	if result.Source.Repo != "test-repo" {
		t.Errorf("Expected Source Repo 'test-repo', got %q", result.Source.Repo)
	}
	if result.Config["model_type"] != "llama" {
		t.Errorf("Expected Config model_type 'llama', got %v", result.Config["model_type"])
	}
	if result.Tensors["layernorm"] != "some-tensor-metadata" {
		t.Errorf("Expected Tensors layernorm 'some-tensor-metadata', got %v", result.Tensors["layernorm"])
	}

	// Verify that the files URL was rewritten correctly to point to the local blob server
	rewrittenFile, exists := result.Files[sha]
	if !exists {
		t.Fatalf("Expected file sha to exist in Files map")
	}
	expectedSource := fmt.Sprintf("http://%s:%d/blobs/%s", routerIP, s.HTTPPort(), sha)
	if rewrittenFile.Source != expectedSource {
		t.Errorf("Expected rewritten file source %q, got %q", expectedSource, rewrittenFile.Source)
	}

	// Verify that the file was indeed cached on disk
	cachedData, err := os.ReadFile(filepath.Join(tempDir, sha))
	if err != nil {
		t.Fatalf("failed to read cached file: %v", err)
	}
	if !bytes.Equal(cachedData, fileContent) {
		t.Errorf("Expected cached file content %q, got %q", string(fileContent), string(cachedData))
	}
}
