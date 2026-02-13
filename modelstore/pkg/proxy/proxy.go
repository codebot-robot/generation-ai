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
	"net/url"
	"os"
	"path/filepath"
	"sync"

	"k8s.io/klog/v2"
)

type Proxy struct {
	upstream *url.URL
	cacheDir string
	mu       sync.Mutex
}

func NewProxy(upstreamURL, cacheDir string) (*Proxy, error) {
	u, err := url.Parse(upstreamURL)
	if err != nil {
		return nil, fmt.Errorf("invalid upstream URL %q: %w", upstreamURL, err)
	}
	return &Proxy{
		upstream: u,
		cacheDir: cacheDir,
	}, nil
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	log := klog.FromContext(ctx)

	// Only cache GET requests
	if r.Method != http.MethodGet {
		p.proxyOnly(w, r)
		return
	}

	cachePath := filepath.Join(p.cacheDir, r.URL.Path)

	// Check if file is in cache
	if _, err := os.Stat(cachePath); err == nil {
		log.V(2).Info("serving from cache", "path", r.URL.Path)
		http.ServeFile(w, r, cachePath)
		return
	}

	// Not in cache, fetch and store
	if err := p.fetchAndStore(w, r, cachePath); err != nil {
		log.Error(err, "failed to fetch and store", "path", r.URL.Path)
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

var noRedirectClient = &http.Client{
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	},
}

func (p *Proxy) proxyOnly(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	log := klog.FromContext(ctx)

	target := *p.upstream
	target.Path = r.URL.Path
	target.RawQuery = r.URL.RawQuery

	req, err := http.NewRequestWithContext(ctx, r.Method, target.String(), r.Body)
	if err != nil {
		log.Error(err, "failed to create request", "url", target.String())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header = r.Header.Clone()
	req.Header.Del("Host")

	resp, err := noRedirectClient.Do(req)
	if err != nil {
		log.Error(err, "failed to proxy request", "url", target.String())
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)
	n, err := io.Copy(w, resp.Body)
	if err != nil {
		log.Error(err, "failed to copy response body", "url", target.String())
	} else {
		log.Info("proxied request", "url", target.String(), "status", resp.StatusCode, "bytes", n)
	}
}

func (p *Proxy) fetchAndStore(w http.ResponseWriter, r *http.Request, cachePath string) error {
	ctx := r.Context()
	log := klog.FromContext(ctx)

	target := *p.upstream
	target.Path = r.URL.Path
	target.RawQuery = r.URL.RawQuery

	log.Info("fetching from upstream", "url", target.String())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return fmt.Errorf("failed to create request for %q: %w", target.String(), err)
	}
	req.Header = r.Header.Clone()
	req.Header.Del("Host")
	// Don't pass through some headers that might interfere with caching or range requests if we don't support them fully
	req.Header.Del("If-None-Match")
	req.Header.Del("If-Modified-Since")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch %q: %w", target.String(), err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Info("upstream returned non-OK status, proxying without caching", "status", resp.StatusCode, "url", target.String())
		// Just proxy non-OK responses without caching
		for k, v := range resp.Header {
			w.Header()[k] = v
		}
		w.WriteHeader(resp.StatusCode)
		_, err := io.Copy(w, resp.Body)
		return err
	}

	// Create parent directories
	if err := os.MkdirAll(filepath.Dir(cachePath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Create temporary file to avoid partial reads from other processes
	tmpFile, err := os.CreateTemp(p.cacheDir, "modelstore-*")
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

	// Set headers for the response to client
	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	// Tee the body to both the client and the temporary file
	mw := io.MultiWriter(w, tmpFile)
	n, err := io.Copy(mw, resp.Body)
	if err != nil {
		return fmt.Errorf("error copying body from %q: %w", target.String(), err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("error closing tmp file: %w", err)
	}

	// Move to final location
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := os.Rename(tmpPath, cachePath); err != nil {
		return fmt.Errorf("error renaming tmp file to %q: %w", cachePath, err)
	}
	moved = true
	log.Info("cached file", "url", target.String(), "path", cachePath, "bytes", n)
	return nil
}
