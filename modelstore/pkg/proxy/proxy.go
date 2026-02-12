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
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sync"
)

type Proxy struct {
	upstream *url.URL
	cacheDir string
	mu       sync.Mutex
}

func NewProxy(upstreamURL, cacheDir string) *Proxy {
	u, err := url.Parse(upstreamURL)
	if err != nil {
		log.Fatalf("Invalid upstream URL: %v", err)
	}
	return &Proxy{
		upstream: u,
		cacheDir: cacheDir,
	}
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Only cache GET requests
	if r.Method != http.MethodGet {
		p.proxyOnly(w, r)
		return
	}

	cachePath := filepath.Join(p.cacheDir, r.URL.Path)

	// Check if file is in cache
	if _, err := os.Stat(cachePath); err == nil {
		log.Printf("Serving from cache: %s", r.URL.Path)
		http.ServeFile(w, r, cachePath)
		return
	}

	// Not in cache, fetch and store
	p.fetchAndStore(w, r, cachePath)
}

func (p *Proxy) proxyOnly(w http.ResponseWriter, r *http.Request) {
	target := *p.upstream
	target.Path = r.URL.Path
	target.RawQuery = r.URL.RawQuery

	req, err := http.NewRequest(r.Method, target.String(), r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header = r.Header

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func (p *Proxy) fetchAndStore(w http.ResponseWriter, r *http.Request, cachePath string) {
	// Ensure only one request fetches a particular path to avoid race conditions
	// In a real system we'd want per-path locking. For simplicity, we'll just go ahead.
	// Actually, let's at least avoid concurrent writes to the same file.

	target := *p.upstream
	target.Path = r.URL.Path
	target.RawQuery = r.URL.RawQuery

	log.Printf("Fetching from upstream: %s", target.String())

	req, err := http.NewRequest(http.MethodGet, target.String(), nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header = r.Header
	// Don't pass through some headers that might interfere with caching or range requests if we don't support them fully
	req.Header.Del("If-None-Match")
	req.Header.Del("If-Modified-Since")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Just proxy non-OK responses without caching
		for k, v := range resp.Header {
			w.Header()[k] = v
		}
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	// Create parent directories
	if err := os.MkdirAll(filepath.Dir(cachePath), 0755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Create temporary file to avoid partial reads from other processes
	tmpFile, err := os.CreateTemp(p.cacheDir, "modelstore-*")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	// Set headers for the response to client
	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	// Tee the body to both the client and the temporary file
	mw := io.MultiWriter(w, tmpFile)
	if _, err := io.Copy(mw, resp.Body); err != nil {
		log.Printf("Error copying body: %v", err)
		return
	}

	if err := tmpFile.Close(); err != nil {
		log.Printf("Error closing tmp file: %v", err)
		return
	}

	// Move to final location
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := os.Rename(tmpFile.Name(), cachePath); err != nil {
		log.Printf("Error renaming tmp file: %v", err)
	}
}
