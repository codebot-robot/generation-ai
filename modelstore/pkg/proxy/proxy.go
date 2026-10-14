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

// Package proxy is modelstore's HTTP API: content-addressed blobs,
// Model objects as the index from file names to blobs, imports from
// the Hugging Face Hub, and a Hub-compatible read API so that
// transformers and vLLM load through it with HF_ENDPOINT set.
package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// DirectURLHeader carries, on blob responses, a URL a client with its
// own access to the backing store can read the blob from directly
// (a gs:// URL for the GCS backend).
const DirectURLHeader = "X-Modelstore-Direct-Url"

type Proxy struct {
	store    blobstore.Store
	importer *importer.Importer
	kube     client.Client

	mu      sync.Mutex
	imports map[string]bool // Model imports running in this process
}

// NewProxy serves blobs from store, imports through imp, and keeps
// Model objects through kube. Call Resume to pick up imports a
// previous process left unfinished.
func NewProxy(store blobstore.Store, imp *importer.Importer, kube client.Client) *Proxy {
	return &Proxy{store: store, importer: imp, kube: kube, imports: map[string]bool{}}
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rw := &responseWriter{ResponseWriter: w}
	err := p.serve(rw, r)
	if err == nil {
		return
	}
	log := klog.FromContext(r.Context())
	if rw.started {
		// Headers are out: the error can only be reported by cutting the
		// body short. Most of these are clients closing a large download
		// early, which is not a server failure.
		log.V(2).Info("response cut short", "path", r.URL.Path, "error", err.Error())
		return
	}
	log.Error(err, "request failed", "path", r.URL.Path, "method", r.Method)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

// responseWriter remembers whether a status line has been written.
type responseWriter struct {
	http.ResponseWriter
	started bool
}

func (w *responseWriter) WriteHeader(code int) {
	w.started = true
	w.ResponseWriter.WriteHeader(code)
}

func (w *responseWriter) Write(b []byte) (int, error) {
	w.started = true
	return w.ResponseWriter.Write(b)
}

func (p *Proxy) serve(w http.ResponseWriter, r *http.Request) error {
	if rest, ok := strings.CutPrefix(r.URL.Path, "/blobs/"); ok {
		sha, action, _ := strings.Cut(rest, "/")
		if !blobstore.ValidSHA(sha) {
			http.Error(w, "blob names are lowercase hex sha256 digests", http.StatusBadRequest)
			return nil
		}
		switch {
		case action == "" && r.Method == http.MethodPut:
			return p.handleBlobPut(w, r, sha)
		case action == "" && (r.Method == http.MethodGet || r.Method == http.MethodHead):
			return p.handleBlobGet(w, r, sha)
		case action == "import" && r.Method == http.MethodPost:
			return p.handleBlobImport(w, r, sha)
		}
		http.NotFound(w, r)
		return nil
	}

	if r.URL.Path == "/models" || strings.HasPrefix(r.URL.Path, "/models/") {
		switch {
		case r.URL.Path == "/models/import" && r.Method == http.MethodPost:
			return p.handleModelImport(w, r)
		case r.Method == http.MethodPost:
			return p.handleModelCreate(w, r)
		case r.Method == http.MethodGet && (r.URL.Path == "/models" || r.URL.Path == "/models/"):
			return p.handleModelList(w, r)
		case r.Method == http.MethodGet:
			return p.handleModelGet(w, r)
		}
	}

	// Hugging Face Hub compatible reads: metadata, then downloads.
	if strings.HasPrefix(r.URL.Path, "/api/models/") {
		return p.handleHFMetadata(w, r)
	}
	if strings.Contains(r.URL.Path, "/resolve/") && (r.Method == http.MethodGet || r.Method == http.MethodHead) {
		return p.handleHFDownload(w, r)
	}

	http.NotFound(w, r)
	return nil
}

// modelName is the Model object name for a Hub repo id: a lowercase
// DNS subdomain (Kubernetes object names), so "Qwen/Qwen3-8B-FP8"
// becomes "qwen-qwen3-8b-fp8". The exact id is kept in spec.source.
func modelName(repoID string) string {
	var b strings.Builder
	for _, c := range strings.ToLower(repoID) {
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '.':
			b.WriteRune(c)
		default: // '/', '_', and anything else
			b.WriteRune('-')
		}
	}
	return strings.Trim(b.String(), "-.")
}

// validModelName returns the Model object name for a repo id, or the
// API server's objection to it.
func validModelName(repoID string) (string, error) {
	name := modelName(repoID)
	if errs := validation.IsDNS1123Subdomain(name); len(errs) > 0 {
		return "", fmt.Errorf("repository %q gives an invalid object name %q: %s", repoID, name, strings.Join(errs, "; "))
	}
	return name, nil
}

func (p *Proxy) getModel(ctx context.Context, name string) (*v1alpha1.Model, error) {
	var model v1alpha1.Model
	if err := p.kube.Get(ctx, client.ObjectKey{Name: name}, &model); err != nil {
		return nil, err
	}
	return &model, nil
}

func (p *Proxy) handleHFMetadata(w http.ResponseWriter, r *http.Request) error {
	repoID := strings.TrimPrefix(r.URL.Path, "/api/models/")
	// /api/models/{repo}/revision/{rev} is the same answer: we hold one revision.
	if i := strings.Index(repoID, "/revision/"); i >= 0 {
		repoID = repoID[:i]
	}
	name, nerr := validModelName(repoID)
	if nerr != nil {
		http.NotFound(w, r)
		return nil
	}
	model, err := p.getModel(r.Context(), name)
	if apierrors.IsNotFound(err) {
		http.NotFound(w, r)
		return nil
	}
	if err != nil {
		return fmt.Errorf("getting model %s: %w", name, err)
	}
	siblings := make([]map[string]any, 0, len(model.Spec.Files))
	for _, f := range model.Spec.Files {
		s := map[string]any{"rfilename": f.Path}
		if f.Size > 0 {
			s["size"] = f.Size
		}
		siblings = append(siblings, s)
	}
	resp := map[string]any{"modelId": repoID, "id": repoID, "siblings": siblings}
	if src := model.Spec.Source; src != nil && src.CommitSHA != "" {
		resp["sha"] = src.CommitSHA
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

func (p *Proxy) handleHFDownload(w http.ResponseWriter, r *http.Request) error {
	before, revAndPath, _ := strings.Cut(r.URL.Path, "/resolve/")
	repoID := strings.TrimPrefix(before, "/")
	_, path, ok := strings.Cut(revAndPath, "/") // the revision is whatever was imported
	if repoID == "" || !ok || path == "" {
		http.NotFound(w, r)
		return nil
	}
	name, nerr := validModelName(repoID)
	if nerr != nil {
		http.NotFound(w, r)
		return nil
	}
	model, err := p.getModel(r.Context(), name)
	if apierrors.IsNotFound(err) {
		http.NotFound(w, r)
		return nil
	}
	if err != nil {
		return fmt.Errorf("getting model %s: %w", name, err)
	}
	for _, f := range model.Spec.Files {
		if f.Path == path {
			// huggingface_hub reads these to decide on caching.
			w.Header().Set("X-Repo-Commit", commitOf(model))
			w.Header().Set("X-Linked-Etag", `"`+f.SHA256+`"`)
			w.Header().Set("ETag", `"`+f.SHA256+`"`)
			return p.serveBlob(w, r, f.SHA256)
		}
	}
	http.NotFound(w, r)
	return nil
}

func commitOf(model *v1alpha1.Model) string {
	if model.Spec.Source != nil && model.Spec.Source.CommitSHA != "" {
		return model.Spec.Source.CommitSHA
	}
	return "main"
}

func (p *Proxy) handleBlobPut(w http.ResponseWriter, r *http.Request, sha string) error {
	log := klog.FromContext(r.Context())
	_, n, err := p.store.Put(r.Context(), sha, r.Body)
	if err != nil {
		if strings.Contains(err.Error(), "sha256 mismatch") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return nil
		}
		return err
	}
	log.Info("stored blob", "sha", sha, "bytes", n)
	w.WriteHeader(http.StatusCreated)
	return nil
}

func (p *Proxy) handleBlobGet(w http.ResponseWriter, r *http.Request, sha string) error {
	return p.serveBlob(w, r, sha)
}

// serveBlob answers GET/HEAD for one blob with Range support (a single
// range: executors read one tensor per request).
func (p *Proxy) serveBlob(w http.ResponseWriter, r *http.Request, sha string) error {
	ctx := r.Context()
	size, err := p.store.Stat(ctx, sha)
	if errors.Is(err, blobstore.ErrNotFound) {
		http.NotFound(w, r)
		return nil
	}
	if err != nil {
		return fmt.Errorf("stat blob %s: %w", sha, err)
	}
	w.Header().Set("Accept-Ranges", "bytes")
	w.Header().Set("Content-Type", "application/octet-stream")
	if u := p.store.DirectURL(sha); u != "" {
		w.Header().Set(DirectURLHeader, u)
	}
	offset, length := int64(0), size
	status := http.StatusOK
	if rng := r.Header.Get("Range"); rng != "" {
		o, l, ok := parseRange(rng, size)
		if !ok {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", size))
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return nil
		}
		offset, length, status = o, l, http.StatusPartialContent
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", offset, offset+length-1, size))
	}
	w.Header().Set("Content-Length", strconv.FormatInt(length, 10))
	if r.Method == http.MethodHead {
		w.WriteHeader(status)
		return nil
	}
	rc, err := p.store.Open(ctx, sha, offset, length)
	if err != nil {
		return fmt.Errorf("open blob %s: %w", sha, err)
	}
	defer func() {
		if err := rc.Close(); err != nil {
			klog.FromContext(ctx).Error(err, "closing blob", "sha", sha)
		}
	}()
	w.WriteHeader(status)
	_, err = io.CopyN(w, rc, length)
	return err
}

// parseRange handles a single byte range: bytes=a-b, bytes=a-, bytes=-n.
// Returns offset, length and whether the range is satisfiable.
func parseRange(h string, size int64) (int64, int64, bool) {
	spec, ok := strings.CutPrefix(strings.TrimSpace(h), "bytes=")
	if !ok || strings.Contains(spec, ",") {
		return 0, 0, false
	}
	first, last, ok := strings.Cut(spec, "-")
	if !ok {
		return 0, 0, false
	}
	first, last = strings.TrimSpace(first), strings.TrimSpace(last)
	if first == "" { // suffix: last n bytes
		n, err := strconv.ParseInt(last, 10, 64)
		if err != nil || n <= 0 {
			return 0, 0, false
		}
		if n > size {
			n = size
		}
		return size - n, n, true
	}
	start, err := strconv.ParseInt(first, 10, 64)
	if err != nil || start < 0 || start >= size {
		return 0, 0, false
	}
	end := size - 1
	if last != "" {
		end, err = strconv.ParseInt(last, 10, 64)
		if err != nil || end < start {
			return 0, 0, false
		}
		if end >= size {
			end = size - 1
		}
	}
	return start, end - start + 1, true
}

// BlobImportRequest asks modelstore to fetch a blob itself.
type BlobImportRequest struct {
	URL string `json:"url"`
}

// handleBlobImport starts (or joins) the import of one blob from a URL.
// 201 when the blob is present, 202 while it is being fetched; the
// client polls HEAD /blobs/<sha> or re-POSTs (both idempotent).
func (p *Proxy) handleBlobImport(w http.ResponseWriter, r *http.Request, sha string) error {
	var req BlobImportRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.URL == "" {
		http.Error(w, `body must be {"url": "..."}`, http.StatusBadRequest)
		return nil
	}
	st, err := p.importer.EnsureBlob(r.Context(), sha, req.URL)
	if errors.Is(err, importer.ErrBadSource) {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil
	}
	if err != nil {
		return err
	}
	if r.URL.Query().Get("wait") == "true" {
		if _, err := p.importer.Wait(r.Context(), sha); err != nil {
			return err
		}
		st = importer.Present
	}
	if st == importer.Present {
		w.WriteHeader(http.StatusCreated)
	} else {
		w.WriteHeader(http.StatusAccepted)
	}
	return nil
}

// ModelImportRequest asks modelstore to import a Hub repository.
type ModelImportRequest struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision,omitempty"`
}

// handleModelImport imports a Hub repository revision: creates the Model
// in phase Importing, streams every file into the store in the
// background, updates status as it goes, and marks it Ready. Returns
// 202 with the Model; watch it with kubectl get models.
func (p *Proxy) handleModelImport(w http.ResponseWriter, r *http.Request) error {
	var req ModelImportRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Repo == "" {
		http.Error(w, `body must be {"repo": "org/name", "revision": "main"}`, http.StatusBadRequest)
		return nil
	}
	ctx := r.Context()
	name, err := validModelName(req.Repo)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil
	}
	model, err := p.getModel(ctx, name)
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("getting model %s: %w", name, err)
	}
	if model != nil && (model.Spec.Source == nil || model.Spec.Source.Repo != req.Repo) {
		// Names are lossy (case folded, punctuation collapsed): never let
		// one repository overwrite a Model that came from another source.
		from := "an upload"
		if model.Spec.Source != nil {
			from = model.Spec.Source.Repo
		}
		http.Error(w, fmt.Sprintf("model %s already exists from %s; cannot import %s under the same name", name, from, req.Repo), http.StatusConflict)
		return nil
	}
	if model != nil && model.Status.Phase == v1alpha1.PhaseImporting && p.running(name) {
		w.WriteHeader(http.StatusAccepted)
		return json.NewEncoder(w).Encode(model)
	}
	if model == nil {
		model = &v1alpha1.Model{ObjectMeta: metav1.ObjectMeta{Name: name}}
		model.Spec.Source = &v1alpha1.ModelSource{Repo: req.Repo, Revision: req.Revision}
		if err := p.kube.Create(ctx, model); err != nil {
			return fmt.Errorf("creating model %s: %w", name, err)
		}
	}
	if err := p.setStatus(ctx, model, v1alpha1.ModelStatus{Phase: v1alpha1.PhaseImporting}); err != nil {
		return err
	}
	p.start(name, req)
	w.WriteHeader(http.StatusAccepted)
	return json.NewEncoder(w).Encode(model)
}

// start runs a Model import in the background unless one is running.
func (p *Proxy) start(name string, req ModelImportRequest) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.imports[name] {
		return
	}
	p.imports[name] = true
	go func() {
		defer func() {
			p.mu.Lock()
			delete(p.imports, name)
			p.mu.Unlock()
		}()
		p.runModelImport(name, req)
	}()
}

// running reports whether this process is importing the Model.
func (p *Proxy) running(name string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.imports[name]
}

// Resume restarts the imports of Models left in phase Importing by a
// previous process (a pod replaced mid-import). Blobs already in the
// store are skipped, so a resumed import costs only what was missing.
func (p *Proxy) Resume(ctx context.Context) error {
	log := klog.FromContext(ctx)
	var list v1alpha1.ModelList
	if err := p.kube.List(ctx, &list); err != nil {
		return fmt.Errorf("listing models: %w", err)
	}
	for i := range list.Items {
		m := &list.Items[i]
		if m.Status.Phase != v1alpha1.PhaseImporting {
			continue
		}
		if m.Spec.Source == nil {
			log.Info("model stuck in Importing with no source; marking Failed", "model", m.Name)
			if err := p.setStatus(ctx, m, v1alpha1.ModelStatus{Phase: v1alpha1.PhaseFailed,
				Message: "import interrupted and no source recorded"}); err != nil {
				log.Error(err, "marking model Failed", "model", m.Name)
			}
			continue
		}
		log.Info("resuming import interrupted by a restart", "model", m.Name, "repo", m.Spec.Source.Repo)
		p.start(m.Name, ModelImportRequest{Repo: m.Spec.Source.Repo, Revision: m.Spec.Source.Revision})
	}
	return nil
}

func (p *Proxy) runModelImport(name string, req ModelImportRequest) {
	ctx := context.Background()
	log := klog.FromContext(ctx).WithValues("model", name, "repo", req.Repo, "revision", req.Revision)
	report := func(prog importer.Progress) {
		// Progress is advisory: a failed status write is logged, not fatal.
		m, err := p.getModel(ctx, name)
		if err == nil {
			err = p.setStatus(ctx, m, v1alpha1.ModelStatus{Phase: v1alpha1.PhaseImporting,
				FilesTotal: prog.FilesTotal, FilesImported: prog.FilesImported, BytesImported: prog.BytesImported})
		}
		if err != nil {
			log.Error(err, "recording import progress")
		}
	}
	commit, files, err := p.importer.ImportRepo(ctx, req.Repo, req.Revision, report)
	// Recording the outcome retries, because leaving the Model in
	// Importing on a transient API error would strand it until a restart.
	record := func() error {
		model, gerr := p.getModel(ctx, name)
		if apierrors.IsNotFound(gerr) {
			log.Info("model deleted during import; dropping the result")
			return nil
		}
		if gerr != nil {
			return gerr
		}
		if err != nil {
			return p.setStatus(ctx, model, v1alpha1.ModelStatus{Phase: v1alpha1.PhaseFailed, Message: err.Error(),
				FilesTotal: model.Status.FilesTotal, FilesImported: model.Status.FilesImported, BytesImported: model.Status.BytesImported})
		}
		model.Spec.Files = files
		model.Spec.Source = &v1alpha1.ModelSource{Repo: req.Repo, Revision: req.Revision, CommitSHA: commit}
		if uerr := p.kube.Update(ctx, model); uerr != nil {
			return uerr
		}
		var bytes int64
		for _, f := range files {
			bytes += f.Size
		}
		return p.setStatus(ctx, model, v1alpha1.ModelStatus{Phase: v1alpha1.PhaseReady,
			FilesTotal: len(files), FilesImported: len(files), BytesImported: bytes})
	}
	var rerr error
	for attempt := 0; attempt < 5; attempt++ {
		if rerr = record(); rerr == nil {
			break
		}
		log.Error(rerr, "recording import outcome; will retry", "attempt", attempt)
		time.Sleep(time.Duration(attempt+1) * time.Second)
	}
	switch {
	case rerr != nil:
		log.Error(rerr, "giving up recording the import outcome; the Model stays Importing until a restart resumes it")
	case err != nil:
		log.Error(err, "import failed")
	default:
		log.Info("imported model", "commit", commit, "files", len(files))
	}
}

func (p *Proxy) setStatus(ctx context.Context, model *v1alpha1.Model, st v1alpha1.ModelStatus) error {
	model.Status = st
	if err := p.kube.Status().Update(ctx, model); err != nil {
		return fmt.Errorf("updating status of %s: %w", model.Name, err)
	}
	return nil
}

func (p *Proxy) handleModelCreate(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	log := klog.FromContext(ctx)
	var model v1alpha1.Model
	if err := json.NewDecoder(r.Body).Decode(&model); err != nil {
		http.Error(w, fmt.Sprintf("decoding model: %v", err), http.StatusBadRequest)
		return nil
	}
	if model.Name == "" {
		http.Error(w, "model name is required", http.StatusBadRequest)
		return nil
	}
	for _, file := range model.Spec.Files {
		if _, err := p.store.Stat(ctx, file.SHA256); err != nil {
			if errors.Is(err, blobstore.ErrNotFound) {
				http.Error(w, fmt.Sprintf("blob %s (path %s) not found", file.SHA256, file.Path), http.StatusBadRequest)
				return nil
			}
			return fmt.Errorf("stat blob %s: %w", file.SHA256, err)
		}
	}
	model.ResourceVersion = ""
	model.UID = ""
	model.CreationTimestamp = metav1.Time{}
	if err := p.kube.Create(ctx, &model); err != nil {
		return fmt.Errorf("creating model: %w", err)
	}
	if err := p.setStatus(ctx, &model, v1alpha1.ModelStatus{Phase: v1alpha1.PhaseReady,
		FilesTotal: len(model.Spec.Files), FilesImported: len(model.Spec.Files)}); err != nil {
		log.Error(err, "setting status", "model", model.Name)
	}
	log.Info("created model", "name", model.Name)
	w.WriteHeader(http.StatusCreated)
	return json.NewEncoder(w).Encode(model)
}

func (p *Proxy) handleModelList(w http.ResponseWriter, r *http.Request) error {
	var list v1alpha1.ModelList
	if err := p.kube.List(r.Context(), &list); err != nil {
		return fmt.Errorf("listing models: %w", err)
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(list)
}

func (p *Proxy) handleModelGet(w http.ResponseWriter, r *http.Request) error {
	name := strings.TrimPrefix(r.URL.Path, "/models/")
	model, err := p.getModel(r.Context(), name)
	if apierrors.IsNotFound(err) {
		http.NotFound(w, r)
		return nil
	}
	if err != nil {
		return fmt.Errorf("getting model %s: %w", name, err)
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(model)
}
