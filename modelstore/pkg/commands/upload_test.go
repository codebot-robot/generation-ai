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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	"github.com/gke-labs/generation-ai/modelstore/pkg/proxy"
	"github.com/gke-labs/generation-ai/modelstore/pkg/registry"
)

func TestBuildUploadCommand(t *testing.T) {
	cmd := BuildUploadCommand()
	if cmd.Use != "upload" {
		t.Errorf("Use = %q", cmd.Use)
	}
	for _, f := range []string{"model-name", "model-dir", "modelstore-url", "tag"} {
		if cmd.Flags().Lookup(f) == nil {
			t.Errorf("flag %s missing", f)
		}
	}
}

func TestRunUpload(t *testing.T) {
	store, err := blobstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	reg := registry.New(store)
	srv := httptest.NewServer(proxy.NewProxy(reg, importer.New(store, "")))
	defer srv.Close()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"a": 1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "weights.bin"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	opt := &UploadOptions{ModelstoreURL: srv.URL, ModelName: "org/test", ModelDir: dir, Tag: "v1"}
	if err := RunUpload(t.Context(), opt); err != nil {
		t.Fatalf("RunUpload: %v", err)
	}
	// Uploading again skips present blobs and rewrites the tag.
	if err := RunUpload(t.Context(), opt); err != nil {
		t.Fatalf("second RunUpload: %v", err)
	}
	resp, err := http.Get(srv.URL + "/v2/org/test/manifests/v1")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	var m registry.Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil || len(m.Layers) != 2 {
		t.Fatalf("manifest: %v %+v", err, m)
	}
	titles := map[string]bool{}
	for _, l := range m.Layers {
		titles[l.Annotations[registry.AnnotationTitle]] = true
	}
	if !titles["config.json"] || !titles["sub/weights.bin"] {
		t.Errorf("layer titles: %v", titles)
	}
	resp2, err := http.Get(srv.URL + "/org/test/resolve/v1/sub/weights.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp2.Body.Close() }()
	if resp2.StatusCode != 200 {
		t.Errorf("resolve after upload: %d", resp2.StatusCode)
	}
	if err := RunUpload(t.Context(), &UploadOptions{ModelstoreURL: srv.URL, ModelName: "Bad/Name", ModelDir: dir, Tag: "v1"}); err == nil {
		t.Error("invalid name should fail before any upload")
	}
}
