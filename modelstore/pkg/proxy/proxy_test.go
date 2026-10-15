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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func newTestProxy(t *testing.T, hfEndpoint string) (*Proxy, client.Client) {
	t.Helper()
	store, err := blobstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	kube := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&v1alpha1.Model{}).Build()
	return NewProxy(store, importer.New(store, hfEndpoint), kube), kube
}

func shaOf(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func do(p *Proxy, method, path string, body io.Reader, headers ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, body)
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	w := httptest.NewRecorder()
	p.ServeHTTP(w, req)
	return w
}

func TestModelName(t *testing.T) {
	for in, want := range map[string]string{
		"Qwen/Qwen3-8B-FP8":                   "qwen-qwen3-8b-fp8",
		"google/gemma-4-E4B-it":               "google-gemma-4-e4b-it",
		"trl-internal-testing/tiny_random.v2": "trl-internal-testing-tiny-random.v2",
		"user/repo":                           "user-repo",
	} {
		if got := modelName(in); got != want {
			t.Errorf("modelName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBlobPutGetAndRanges(t *testing.T) {
	p, _ := newTestProxy(t, "")
	content := []byte("0123456789abcdef")
	sha := shaOf(content)

	if w := do(p, http.MethodPut, "/blobs/"+sha, bytes.NewReader(content)); w.Code != http.StatusCreated {
		t.Fatalf("PUT: got %d: %s", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodPut, "/blobs/"+strings.Repeat("0", 64), bytes.NewReader(content)); w.Code != http.StatusBadRequest {
		t.Errorf("PUT with wrong sha: got %d, want 400", w.Code)
	}
	if w := do(p, http.MethodPut, "/blobs/not-a-sha", bytes.NewReader(content)); w.Code != http.StatusBadRequest {
		t.Errorf("PUT with malformed name: got %d, want 400", w.Code)
	}

	w := do(p, http.MethodHead, "/blobs/"+sha, nil)
	if w.Code != http.StatusOK || w.Header().Get("Content-Length") != "16" || w.Header().Get("Accept-Ranges") != "bytes" {
		t.Errorf("HEAD: got %d %v", w.Code, w.Header())
	}
	if w := do(p, http.MethodGet, "/blobs/"+sha, nil); w.Code != http.StatusOK || w.Body.String() != string(content) {
		t.Errorf("GET: got %d %q", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodGet, "/blobs/"+strings.Repeat("1", 64), nil); w.Code != http.StatusNotFound {
		t.Errorf("GET missing: got %d, want 404", w.Code)
	}

	for _, tc := range []struct {
		rng, want, contentRange string
	}{
		{"bytes=4-7", "4567", "bytes 4-7/16"},
		{"bytes=12-", "cdef", "bytes 12-15/16"},
		{"bytes=-3", "def", "bytes 13-15/16"},
		{"bytes=10-99", "abcdef", "bytes 10-15/16"},
	} {
		w := do(p, http.MethodGet, "/blobs/"+sha, nil, "Range", tc.rng)
		if w.Code != http.StatusPartialContent || w.Body.String() != tc.want ||
			w.Header().Get("Content-Range") != tc.contentRange ||
			w.Header().Get("Content-Length") != fmt.Sprint(len(tc.want)) {
			t.Errorf("Range %s: got %d %q %q", tc.rng, w.Code, w.Body.String(), w.Header().Get("Content-Range"))
		}
	}
	if w := do(p, http.MethodGet, "/blobs/"+sha, nil, "Range", "bytes=16-20"); w.Code != http.StatusRequestedRangeNotSatisfiable {
		t.Errorf("unsatisfiable range: got %d, want 416", w.Code)
	}
	if w := do(p, http.MethodGet, "/blobs/"+sha, nil, "Range", "bytes=0-1,4-5"); w.Code != http.StatusRequestedRangeNotSatisfiable {
		t.Errorf("multi range: got %d, want 416", w.Code)
	}
}

func TestModelCreateAndHubReads(t *testing.T) {
	p, _ := newTestProxy(t, "")
	content := []byte("blob content")
	sha := shaOf(content)
	do(p, http.MethodPut, "/blobs/"+sha, bytes.NewReader(content))

	missing := strings.Repeat("2", 64)
	bad, _ := json.Marshal(v1alpha1.Model{ObjectMeta: metav1.ObjectMeta{Name: "bad"},
		Spec: v1alpha1.ModelSpec{Files: []v1alpha1.File{{Path: "x", SHA256: missing}}}})
	if w := do(p, http.MethodPost, "/models", bytes.NewReader(bad)); w.Code != http.StatusBadRequest {
		t.Errorf("model with missing blob: got %d, want 400", w.Code)
	}

	model, _ := json.Marshal(v1alpha1.Model{ObjectMeta: metav1.ObjectMeta{Name: "user-repo"},
		Spec: v1alpha1.ModelSpec{Files: []v1alpha1.File{{Path: "config.json", SHA256: sha, Size: int64(len(content))}}}})
	if w := do(p, http.MethodPost, "/models", bytes.NewReader(model)); w.Code != http.StatusCreated {
		t.Fatalf("POST model: got %d: %s", w.Code, w.Body.String())
	}
	w := do(p, http.MethodGet, "/models/user-repo", nil)
	var got v1alpha1.Model
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil || got.Status.Phase != v1alpha1.PhaseReady {
		t.Errorf("GET model: %v phase=%q", err, got.Status.Phase)
	}
	w = do(p, http.MethodGet, "/models", nil)
	var list v1alpha1.ModelList
	if err := json.NewDecoder(w.Body).Decode(&list); err != nil || len(list.Items) != 1 {
		t.Errorf("list: %v %d items", err, len(list.Items))
	}

	// Hub-compatible reads, with and without a revision in the path.
	for _, path := range []string{"/api/models/user/repo", "/api/models/user/repo/revision/main"} {
		w := do(p, http.MethodGet, path, nil)
		var meta map[string]any
		if err := json.NewDecoder(w.Body).Decode(&meta); w.Code != 200 || err != nil || meta["modelId"] != "user/repo" {
			t.Errorf("%s: %d %v %v", path, w.Code, err, meta)
		}
	}
	w = do(p, http.MethodGet, "/user/repo/resolve/main/config.json", nil)
	if w.Code != http.StatusOK || w.Body.String() != string(content) || w.Header().Get("ETag") != `"`+sha+`"` {
		t.Errorf("resolve: %d %q etag=%s", w.Code, w.Body.String(), w.Header().Get("ETag"))
	}
	w = do(p, http.MethodGet, "/user/repo/resolve/main/config.json", nil, "Range", "bytes=0-3")
	if w.Code != http.StatusPartialContent || w.Body.String() != "blob" {
		t.Errorf("resolve range: %d %q", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodGet, "/user/repo/resolve/main/missing.bin", nil); w.Code != http.StatusNotFound {
		t.Errorf("resolve missing file: %d", w.Code)
	}
	if w := do(p, http.MethodGet, "/api/models/nobody/nothing", nil); w.Code != http.StatusNotFound {
		t.Errorf("metadata of unknown model: %d", w.Code)
	}
}

// fakeHub serves a one-repo Hugging Face Hub: a listing with one LFS
// file (sha256 published) and one small file (no sha256).
func fakeHub(t *testing.T, weights, config []byte) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/name/revision/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"sha": "abc123",
			"siblings": []map[string]any{
				{"rfilename": "model.safetensors", "size": len(weights),
					"lfs": map[string]any{"sha256": shaOf(weights), "size": len(weights)}},
				{"rfilename": "config.json", "size": len(config)},
			},
		})
	})
	mux.HandleFunc("/org/name/resolve/abc123/model.safetensors", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(weights) })
	mux.HandleFunc("/org/name/resolve/abc123/config.json", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(config) })
	return httptest.NewServer(mux)
}

func TestBlobImport(t *testing.T) {
	weights := bytes.Repeat([]byte("w"), 1000)
	hub := fakeHub(t, weights, []byte("{}"))
	defer hub.Close()
	p, _ := newTestProxy(t, hub.URL)
	sha := shaOf(weights)
	body := fmt.Sprintf(`{"url": %q}`, hub.URL+"/org/name/resolve/abc123/model.safetensors")

	w := do(p, http.MethodPost, "/blobs/"+sha+"/import?wait=true", strings.NewReader(body))
	if w.Code != http.StatusCreated {
		t.Fatalf("import: got %d: %s", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodGet, "/blobs/"+sha, nil, "Range", "bytes=0-9"); w.Code != 206 || w.Body.String() != "wwwwwwwwww" {
		t.Errorf("after import: %d %q", w.Code, w.Body.String())
	}
	// Importing again is a no-op that reports the blob present.
	if w := do(p, http.MethodPost, "/blobs/"+sha+"/import", strings.NewReader(body)); w.Code != http.StatusCreated {
		t.Errorf("re-import: got %d, want 201", w.Code)
	}
	// A URL whose bytes do not match the sha leaves no blob.
	wrong := strings.Repeat("3", 64)
	w = do(p, http.MethodPost, "/blobs/"+wrong+"/import?wait=true", strings.NewReader(body))
	if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "sha256 mismatch") {
		t.Errorf("mismatched import: %d %s", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodHead, "/blobs/"+wrong, nil); w.Code != http.StatusNotFound {
		t.Errorf("mismatched blob should not exist: %d", w.Code)
	}
}

func TestModelImport(t *testing.T) {
	weights := bytes.Repeat([]byte("x"), 4096)
	config := []byte(`{"model_type": "test"}`)
	hub := fakeHub(t, weights, config)
	defer hub.Close()
	p, kube := newTestProxy(t, hub.URL)

	w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name", "revision": "main"}`))
	if w.Code != http.StatusAccepted {
		t.Fatalf("import: got %d: %s", w.Code, w.Body.String())
	}
	var model v1alpha1.Model
	deadline := time.Now().Add(10 * time.Second)
	for {
		if err := kube.Get(t.Context(), client.ObjectKey{Name: "org-name"}, &model); err != nil {
			t.Fatal(err)
		}
		if model.Status.Phase != v1alpha1.PhaseImporting || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if model.Status.Phase != v1alpha1.PhaseReady {
		t.Fatalf("phase %q (%s)", model.Status.Phase, model.Status.Message)
	}
	if model.Spec.Source == nil || model.Spec.Source.CommitSHA != "abc123" || len(model.Spec.Files) != 2 {
		t.Fatalf("spec: %+v", model.Spec)
	}
	if model.Status.FilesImported != 2 || model.Status.BytesImported != int64(len(weights)+len(config)) {
		t.Errorf("status: %+v", model.Status)
	}
	// The small file was hashed on the way in; the Hub reads now serve both.
	w = do(p, http.MethodGet, "/org/name/resolve/main/config.json", nil)
	if w.Code != 200 || w.Body.String() != string(config) || w.Header().Get("X-Repo-Commit") != "abc123" {
		t.Errorf("config via resolve: %d %q commit=%s", w.Code, w.Body.String(), w.Header().Get("X-Repo-Commit"))
	}
	w = do(p, http.MethodHead, "/org/name/resolve/main/model.safetensors", nil)
	if w.Code != 200 || w.Header().Get("Content-Length") != "4096" {
		t.Errorf("weights HEAD: %d %s", w.Code, w.Header().Get("Content-Length"))
	}
	w = do(p, http.MethodGet, "/api/models/org/name", nil)
	var meta map[string]any
	_ = json.NewDecoder(w.Body).Decode(&meta)
	if meta["sha"] != "abc123" {
		t.Errorf("metadata sha: %v", meta["sha"])
	}
}

func TestModelImportRefusesCollisionsAndBadNames(t *testing.T) {
	weights := bytes.Repeat([]byte("y"), 100)
	hub := fakeHub(t, weights, []byte("{}"))
	defer hub.Close()
	p, kube := newTestProxy(t, hub.URL)

	// A hand-created Model owns its name; a repository that normalizes to it is refused.
	sha := shaOf(weights)
	do(p, http.MethodPut, "/blobs/"+sha, bytes.NewReader(weights))
	owned, _ := json.Marshal(v1alpha1.Model{ObjectMeta: metav1.ObjectMeta{Name: "org-name"},
		Spec: v1alpha1.ModelSpec{Files: []v1alpha1.File{{Path: "w", SHA256: sha}}}})
	if w := do(p, http.MethodPost, "/models", bytes.NewReader(owned)); w.Code != http.StatusCreated {
		t.Fatalf("create: %d %s", w.Code, w.Body.String())
	}
	w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name"}`))
	if w.Code != http.StatusConflict {
		t.Errorf("import over an upload: got %d, want 409: %s", w.Code, w.Body.String())
	}
	// Two repositories with one normalized name: the second is refused.
	var m v1alpha1.Model
	_ = kube.Get(t.Context(), client.ObjectKey{Name: "org-name"}, &m)
	m.Spec.Source = &v1alpha1.ModelSource{Repo: "Org/Name"}
	_ = kube.Update(t.Context(), &m)
	w = do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name"}`))
	if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "Org/Name") {
		t.Errorf("import over another repo: got %d, want 409 naming the owner: %s", w.Code, w.Body.String())
	}
	// Names the API server would refuse are a client error, not a 500.
	for _, repo := range []string{"org/v1.-beta", "/", "a..b"} {
		w := do(p, http.MethodPost, "/models/import", strings.NewReader(fmt.Sprintf(`{"repo": %q}`, repo)))
		if w.Code != http.StatusBadRequest {
			t.Errorf("repo %q: got %d, want 400: %s", repo, w.Code, w.Body.String())
		}
	}
	if w := do(p, http.MethodGet, "/api/models/", nil); w.Code != http.StatusNotFound {
		t.Errorf("empty repo metadata: got %d, want 404", w.Code)
	}
}

func TestBlobImportRefusesForeignURLs(t *testing.T) {
	hub := fakeHub(t, []byte("w"), []byte("{}"))
	defer hub.Close()
	p, _ := newTestProxy(t, hub.URL)
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("modelstore fetched %s from a host that is not the Hub", r.URL)
	}))
	defer other.Close()
	sha := strings.Repeat("a", 64)
	for _, u := range []string{other.URL + "/x", "https://" + strings.TrimPrefix(hub.URL, "http://") + "/x", "file:///etc/passwd", "::bad"} {
		w := do(p, http.MethodPost, "/blobs/"+sha+"/import", strings.NewReader(fmt.Sprintf(`{"url": %q}`, u)))
		if w.Code != http.StatusBadRequest {
			t.Errorf("url %q: got %d, want 400: %s", u, w.Code, w.Body.String())
		}
	}
}

func TestModelImportEscapesFileNames(t *testing.T) {
	// A fake Hub whose small file has URL syntax in its name.
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/name/revision/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"sha": "c1",
			"siblings": []map[string]any{{"rfilename": "notes#1.md", "size": 5}, {"rfilename": "100%.json", "size": 2}}})
	})
	mux.HandleFunc("/org/name/resolve/c1/notes", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("WRONG")) })
	mux.HandleFunc("/org/name/resolve/c1/notes%231.md", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("right")) })
	mux.HandleFunc("/org/name/resolve/c1/100%25.json", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("{}")) })
	hub := httptest.NewServer(mux)
	defer hub.Close()
	p, kube := newTestProxy(t, hub.URL)
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name"}`)); w.Code != http.StatusAccepted {
		t.Fatalf("import: %d %s", w.Code, w.Body.String())
	}
	model := waitPhase(t, kube, "org-name")
	if model.Status.Phase != v1alpha1.PhaseReady {
		t.Fatalf("phase %q: %s", model.Status.Phase, model.Status.Message)
	}
	w := do(p, http.MethodGet, "/org/name/resolve/main/notes%231.md", nil)
	if w.Code != 200 || w.Body.String() != "right" {
		t.Errorf("escaped file: %d %q", w.Code, w.Body.String())
	}
}

func TestResumeRestartsInterruptedImports(t *testing.T) {
	weights := bytes.Repeat([]byte("z"), 300)
	hub := fakeHub(t, weights, []byte("{}"))
	defer hub.Close()
	p, kube := newTestProxy(t, hub.URL)
	// A Model a previous process left in Importing, and one with no source.
	for _, m := range []v1alpha1.Model{
		{ObjectMeta: metav1.ObjectMeta{Name: "org-name"}, Spec: v1alpha1.ModelSpec{Source: &v1alpha1.ModelSource{Repo: "org/name", Revision: "main"}}},
		{ObjectMeta: metav1.ObjectMeta{Name: "orphan"}},
	} {
		m := m
		if err := kube.Create(t.Context(), &m); err != nil {
			t.Fatal(err)
		}
		m.Status.Phase = v1alpha1.PhaseImporting
		if err := kube.Status().Update(t.Context(), &m); err != nil {
			t.Fatal(err)
		}
	}
	if err := p.Resume(t.Context()); err != nil {
		t.Fatal(err)
	}
	if model := waitPhase(t, kube, "org-name"); model.Status.Phase != v1alpha1.PhaseReady || len(model.Spec.Files) != 2 {
		t.Errorf("resumed import: phase %q files %d (%s)", model.Status.Phase, len(model.Spec.Files), model.Status.Message)
	}
	var orphan v1alpha1.Model
	_ = kube.Get(t.Context(), client.ObjectKey{Name: "orphan"}, &orphan)
	if orphan.Status.Phase != v1alpha1.PhaseFailed {
		t.Errorf("orphan: phase %q, want Failed", orphan.Status.Phase)
	}
}

// waitPhase polls until the Model leaves Importing (or 10 s pass).
func waitPhase(t *testing.T, kube client.Client, name string) v1alpha1.Model {
	t.Helper()
	var model v1alpha1.Model
	deadline := time.Now().Add(10 * time.Second)
	for {
		if err := kube.Get(t.Context(), client.ObjectKey{Name: name}, &model); err != nil {
			t.Fatal(err)
		}
		if model.Status.Phase != v1alpha1.PhaseImporting || time.Now().After(deadline) {
			return model
		}
		time.Sleep(20 * time.Millisecond)
	}
}
