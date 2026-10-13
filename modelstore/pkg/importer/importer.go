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

// Package importer brings files into the blob store from the Hugging
// Face Hub (or any URL) without the bytes passing through a client:
// modelstore streams them itself, verifying the sha256 on the way.
package importer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"k8s.io/klog/v2"
)

// Importer streams blobs into a Store, de-duplicating concurrent
// requests for the same blob.
type Importer struct {
	store      blobstore.Store
	hfEndpoint string
	hfToken    string
	client     *http.Client

	mu   sync.Mutex
	jobs map[string]*job // by sha
}

type job struct {
	done chan struct{}
	err  error
	n    int64
}

// New creates an Importer. hfEndpoint is the Hub base URL (default
// https://huggingface.co); the HF_TOKEN environment variable, when set,
// is sent to it for gated repositories.
func New(store blobstore.Store, hfEndpoint string) *Importer {
	if hfEndpoint == "" {
		hfEndpoint = "https://huggingface.co"
	}
	return &Importer{
		store:      store,
		hfEndpoint: strings.TrimSuffix(hfEndpoint, "/"),
		hfToken:    os.Getenv("HF_TOKEN"),
		client:     &http.Client{Timeout: 0}, // multi-GB downloads; cancellation is by context
		jobs:       map[string]*job{},
	}
}

// Status of a blob: Present with its size, InFlight, or Absent.
type Status int

const (
	Absent Status = iota
	InFlight
	Present
)

// Stat reports whether a blob is present, being imported, or absent.
func (im *Importer) Stat(ctx context.Context, sha string) (Status, int64, error) {
	if size, err := im.store.Stat(ctx, sha); err == nil {
		return Present, size, nil
	} else if !errors.Is(err, blobstore.ErrNotFound) {
		return Absent, 0, err
	}
	im.mu.Lock()
	_, running := im.jobs[sha]
	im.mu.Unlock()
	if running {
		return InFlight, 0, nil
	}
	return Absent, 0, nil
}

// EnsureBlob makes sure blob sha is in the store, fetching it from
// srcURL if needed. It returns as soon as the blob is present or its
// import has started; Wait blocks for completion. Idempotent: a second
// call for a blob in flight joins the existing import.
func (im *Importer) EnsureBlob(ctx context.Context, sha, srcURL string) (Status, error) {
	if !blobstore.ValidSHA(sha) {
		return Absent, fmt.Errorf("invalid sha256 %q", sha)
	}
	st, _, err := im.Stat(ctx, sha)
	if err != nil || st != Absent {
		return st, err
	}
	im.mu.Lock()
	if _, running := im.jobs[sha]; running {
		im.mu.Unlock()
		return InFlight, nil
	}
	j := &job{done: make(chan struct{})}
	im.jobs[sha] = j
	im.mu.Unlock()
	go im.run(sha, srcURL, j)
	return InFlight, nil
}

// Wait blocks until the import of sha (if any) finishes and returns
// its error. Present blobs return immediately.
func (im *Importer) Wait(ctx context.Context, sha string) (int64, error) {
	im.mu.Lock()
	j, running := im.jobs[sha]
	im.mu.Unlock()
	if !running {
		size, err := im.store.Stat(ctx, sha)
		return size, err
	}
	select {
	case <-j.done:
		return j.n, j.err
	case <-ctx.Done():
		return 0, ctx.Err()
	}
}

func (im *Importer) run(sha, srcURL string, j *job) {
	// The import outlives the request that started it: a dropped client
	// must not abort a 50 GB download that a retry would only restart.
	ctx := context.Background()
	log := klog.FromContext(ctx).WithValues("sha", sha, "url", srcURL)
	start := time.Now()
	_, n, err := im.fetch(ctx, sha, srcURL)
	j.n, j.err = n, err
	if err != nil {
		log.Error(err, "import failed", "bytes", n)
	} else {
		log.Info("imported blob", "bytes", n, "seconds", int(time.Since(start).Seconds()))
	}
	im.mu.Lock()
	delete(im.jobs, sha)
	im.mu.Unlock()
	close(j.done)
}

// fetch streams srcURL into the store as blob sha ("" names it by content).
func (im *Importer) fetch(ctx context.Context, sha, srcURL string) (string, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srcURL, nil)
	if err != nil {
		return "", 0, err
	}
	if im.hfToken != "" && im.isHub(srcURL) {
		req.Header.Set("Authorization", "Bearer "+im.hfToken)
	}
	resp, err := im.client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("fetching %s: %w", srcURL, err)
	}
	defer closeBody(ctx, resp, srcURL)
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("fetching %s: %s", srcURL, resp.Status)
	}
	got, n, err := im.store.Put(ctx, sha, resp.Body)
	if err != nil {
		return got, n, fmt.Errorf("storing %s: %w", srcURL, err)
	}
	return got, n, nil
}

func closeBody(ctx context.Context, resp *http.Response, url string) {
	if err := resp.Body.Close(); err != nil {
		klog.FromContext(ctx).Error(err, "closing response body", "url", url)
	}
}

func (im *Importer) isHub(srcURL string) bool {
	u, err := url.Parse(srcURL)
	if err != nil {
		return false
	}
	h, err := url.Parse(im.hfEndpoint)
	return err == nil && u.Host == h.Host
}

// HubFile is one file of a Hub repository at a commit.
type HubFile struct {
	Path   string
	SHA256 string // empty for files the Hub stores outside LFS
	Size   int64
	URL    string
}

// ListRepo resolves a Hub repository revision to its commit and files.
func (im *Importer) ListRepo(ctx context.Context, repo, revision string) (string, []HubFile, error) {
	if revision == "" {
		revision = "main"
	}
	api := fmt.Sprintf("%s/api/models/%s/revision/%s?blobs=true", im.hfEndpoint, repo, url.PathEscape(revision))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, api, nil)
	if err != nil {
		return "", nil, err
	}
	if im.hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+im.hfToken)
	}
	resp, err := im.client.Do(req)
	if err != nil {
		return "", nil, fmt.Errorf("listing %s@%s: %w", repo, revision, err)
	}
	defer closeBody(ctx, resp, api)
	if resp.StatusCode != http.StatusOK {
		body, rerr := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if rerr != nil {
			body = []byte("(unreadable body: " + rerr.Error() + ")")
		}
		return "", nil, fmt.Errorf("listing %s@%s: %s: %s", repo, revision, resp.Status, strings.TrimSpace(string(body)))
	}
	var info struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Rfilename string `json:"rfilename"`
			Size      int64  `json:"size"`
			LFS       *struct {
				SHA256 string `json:"sha256"`
				Size   int64  `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", nil, fmt.Errorf("decoding listing of %s@%s: %w", repo, revision, err)
	}
	commit := info.SHA
	if commit == "" {
		commit = revision
	}
	var files []HubFile
	for _, s := range info.Siblings {
		f := HubFile{Path: s.Rfilename, Size: s.Size,
			URL: fmt.Sprintf("%s/%s/resolve/%s/%s", im.hfEndpoint, repo, commit, s.Rfilename)}
		if s.LFS != nil {
			f.SHA256, f.Size = s.LFS.SHA256, s.LFS.Size
		}
		files = append(files, f)
	}
	return commit, files, nil
}

// Progress is reported by ImportRepo after each file.
type Progress struct {
	FilesTotal, FilesImported int
	BytesImported             int64
}

// ImportRepo brings every file of a Hub repository revision into the
// store and returns the Model files (with digests) for it. LFS files
// are fetched by their published sha256 and verified; other files
// (configs, tokenizers) are hashed as they stream. report, if set, is
// called after each file.
func (im *Importer) ImportRepo(ctx context.Context, repo, revision string, report func(Progress)) (string, []v1alpha1.File, error) {
	commit, files, err := im.ListRepo(ctx, repo, revision)
	if err != nil {
		return "", nil, err
	}
	var out []v1alpha1.File
	prog := Progress{FilesTotal: len(files)}
	for _, f := range files {
		sha := f.SHA256
		if sha != "" {
			if _, err := im.EnsureBlob(ctx, sha, f.URL); err != nil {
				return commit, out, err
			}
			n, err := im.Wait(ctx, sha)
			if err != nil {
				return commit, out, fmt.Errorf("%s: %w", f.Path, err)
			}
			prog.BytesImported += n
		} else {
			got, n, err := im.fetch(ctx, "", f.URL)
			if err != nil {
				return commit, out, fmt.Errorf("%s: %w", f.Path, err)
			}
			sha = got
			prog.BytesImported += n
		}
		out = append(out, v1alpha1.File{Path: f.Path, SHA256: sha, Size: f.Size})
		prog.FilesImported++
		if report != nil {
			report(prog)
		}
	}
	return commit, out, nil
}
