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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type Proxy struct {
	cacheDir string
	mu       sync.Mutex
	kube     client.Client
}

func NewProxy(ignoredUpstreamURL, cacheDir string, kube client.Client) (*Proxy, error) {
	return &Proxy{
		cacheDir: cacheDir,
		kube:     kube,
	}, nil
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := p.serve(w, r); err != nil {
		ctx := r.Context()
		log := klog.FromContext(ctx)
		log.Error(err, "request failed", "path", r.URL.Path, "method", r.Method)
		if !p.isResponseStarted(w) {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func (p *Proxy) isResponseStarted(w http.ResponseWriter) bool {
	return false
}

func (p *Proxy) serve(w http.ResponseWriter, r *http.Request) error {
	if strings.HasPrefix(r.URL.Path, "/blobs/") {
		if r.Method == http.MethodPut {
			return p.handleBlobPut(w, r)
		}
		if r.Method == http.MethodGet || r.Method == http.MethodHead {
			return p.handleBlobGet(w, r)
		}
	}

	if r.URL.Path == "/models" || strings.HasPrefix(r.URL.Path, "/models/") {
		if r.Method == http.MethodPost {
			return p.handleModelCreate(w, r)
		}
		if r.Method == http.MethodGet {
			if r.URL.Path == "/models" || r.URL.Path == "/models/" {
				return p.handleModelList(w, r)
			}
			return p.handleModelGet(w, r)
		}
	}

	// Handle HF metadata API: /api/models/{model_id}
	if strings.HasPrefix(r.URL.Path, "/api/models/") {
		return p.handleHFMetadata(w, r)
	}

	// Handle HF download API: /{model_id}/resolve/{revision}/{path}
	if strings.Contains(r.URL.Path, "/resolve/") {
		return p.handleHFDownload(w, r)
	}

	// For legacy compatibility or direct access to cached files by path
	if r.Method == http.MethodGet {
		cachePath := filepath.Join(p.cacheDir, r.URL.Path)
		if _, err := os.Stat(cachePath); err == nil {
			http.ServeFile(w, r, cachePath)
			return nil
		}
	}

	http.NotFound(w, r)
	return nil
}

func (p *Proxy) handleHFMetadata(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	repoID := strings.TrimPrefix(r.URL.Path, "/api/models/")
	modelName := strings.ReplaceAll(repoID, "/", "-")

	var model v1alpha1.Model
	if err := p.kube.Get(ctx, client.ObjectKey{Name: modelName}, &model); err != nil {
		if apierrors.IsNotFound(err) {
			http.NotFound(w, r)
			return nil
		}
		return fmt.Errorf("failed to get model %s: %w", modelName, err)
	}

	// Return a minimal HF-compatible metadata response
	resp := map[string]any{
		"modelId": repoID,
		"id":      repoID,
		"siblings": func() []map[string]string {
			var siblings []map[string]string
			for _, f := range model.Spec.Files {
				siblings = append(siblings, map[string]string{"rfilename": f.Path})
			}
			return siblings
		}(),
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

func (p *Proxy) handleHFDownload(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	parts := strings.Split(r.URL.Path, "/resolve/")
	if len(parts) != 2 {
		http.NotFound(w, r)
		return nil
	}

	repoID := strings.TrimPrefix(parts[0], "/")
	modelName := strings.ReplaceAll(repoID, "/", "-")

	// revisionAndPath is like "main/config.json"
	revAndPath := parts[1]
	revParts := strings.Split(revAndPath, "/")
	if len(revParts) < 2 {
		http.NotFound(w, r)
		return nil
	}
	// revision := revParts[0] // currently ignored
	path := strings.Join(revParts[1:], "/")

	var model v1alpha1.Model
	if err := p.kube.Get(ctx, client.ObjectKey{Name: modelName}, &model); err != nil {
		if apierrors.IsNotFound(err) {
			http.NotFound(w, r)
			return nil
		}
		return fmt.Errorf("failed to get model %s: %w", modelName, err)
	}

	var sha string
	for _, f := range model.Spec.Files {
		if f.Path == path {
			sha = f.SHA256
			break
		}
	}

	if sha == "" {
		http.NotFound(w, r)
		return nil
	}

	blobPath := filepath.Join(p.cacheDir, "blobs", sha)
	if _, err := os.Stat(blobPath); err != nil {
		return fmt.Errorf("blob %s not found in cache: %w", sha, err)
	}

	http.ServeFile(w, r, blobPath)
	return nil
}

func (p *Proxy) handleBlobPut(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	log := klog.FromContext(ctx)

	sha := strings.TrimPrefix(r.URL.Path, "/blobs/")
	if sha == "" {
		return fmt.Errorf("missing sha in blob put")
	}

	// Create blobs directory if it doesn't exist
	blobsDir := filepath.Join(p.cacheDir, "blobs")
	if err := os.MkdirAll(blobsDir, 0755); err != nil {
		return fmt.Errorf("failed to create blobs directory: %w", err)
	}

	blobPath := filepath.Join(blobsDir, sha)

	// Create temporary file in the same directory to handle concurrent uploads and partial writes
	tmpFile, err := os.CreateTemp(blobsDir, "upload-*")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	moved := false
	defer func() {
		tmpFile.Close()
		if !moved {
			if err := os.Remove(tmpPath); err != nil && !os.IsNotExist(err) {
				log.Error(err, "failed to remove temp file", "path", tmpPath)
			}
		}
	}()

	hasher := sha256.New()
	mw := io.MultiWriter(tmpFile, hasher)

	n, err := io.Copy(mw, r.Body)
	if err != nil {
		return fmt.Errorf("failed to save blob: %w", err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	actualSha := hex.EncodeToString(hasher.Sum(nil))
	if actualSha != sha {
		return fmt.Errorf("sha256 mismatch: expected %s, got %s", sha, actualSha)
	}

	// Move to final location
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := os.Rename(tmpPath, blobPath); err != nil {
		return fmt.Errorf("failed to rename blob: %w", err)
	}
	moved = true

	log.Info("stored blob", "sha", sha, "bytes", n)
	w.WriteHeader(http.StatusCreated)
	return nil
}

func (p *Proxy) handleBlobGet(w http.ResponseWriter, r *http.Request) error {
	sha := strings.TrimPrefix(r.URL.Path, "/blobs/")
	if sha == "" {
		return fmt.Errorf("missing sha in blob get")
	}

	blobPath := filepath.Join(p.cacheDir, "blobs", sha)
	if _, err := os.Stat(blobPath); err != nil {
		if os.IsNotExist(err) {
			http.NotFound(w, r)
			return nil
		}
		return fmt.Errorf("failed to check blob: %w", err)
	}

	if r.Method == http.MethodHead {
		return nil
	}

	http.ServeFile(w, r, blobPath)
	return nil
}

func (p *Proxy) handleModelCreate(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	log := klog.FromContext(ctx)

	var model v1alpha1.Model
	if err := json.NewDecoder(r.Body).Decode(&model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}

	if model.Name == "" {
		return fmt.Errorf("model name is required")
	}

	// Verify all blobs exist
	for _, file := range model.Spec.Files {
		blobPath := filepath.Join(p.cacheDir, "blobs", file.SHA256)
		if _, err := os.Stat(blobPath); err != nil {
			return fmt.Errorf("blob %s (path %s) not found: %w", file.SHA256, file.Path, err)
		}
	}

	model.ResourceVersion = ""
	model.UID = ""
	model.CreationTimestamp = metav1.Time{}

	if err := p.kube.Create(ctx, &model); err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	log.Info("created model", "name", model.Name)
	w.WriteHeader(http.StatusCreated)
	return json.NewEncoder(w).Encode(model)
}

func (p *Proxy) handleModelList(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	var list v1alpha1.ModelList
	if err := p.kube.List(ctx, &list); err != nil {
		return fmt.Errorf("failed to list models: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(list)
}

func (p *Proxy) handleModelGet(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	modelName := strings.TrimPrefix(r.URL.Path, "/models/")

	var model v1alpha1.Model
	if err := p.kube.Get(ctx, client.ObjectKey{Name: modelName}, &model); err != nil {
		if apierrors.IsNotFound(err) {
			http.NotFound(w, r)
			return nil
		}
		return fmt.Errorf("failed to get model %s: %w", modelName, err)
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(model)
}
