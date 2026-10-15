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

package importer

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
)

func TestStalledDownloadIsCancelled(t *testing.T) {
	// A peer that sends headers and a few bytes, then goes silent.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("partial"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	store, err := blobstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	old := StallTimeout
	StallTimeout = 300 * time.Millisecond
	defer func() { StallTimeout = old }()

	im := New(store, srv.URL)
	sha := strings.Repeat("a", 64)
	if _, err := im.EnsureBlob(t.Context(), sha, srv.URL+"/stall"); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := im.Wait(t.Context(), sha); done <- err }()
	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "no data for") {
			t.Fatalf("want a stall error, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("stalled download was not cancelled")
	}
	if st, _, _ := im.Stat(t.Context(), sha); st != Absent {
		t.Errorf("after a failed import the blob should be absent, got %v", st)
	}
}

func TestCheckSource(t *testing.T) {
	im := New(nil, "https://huggingface.co")
	for u, ok := range map[string]bool{
		"https://huggingface.co/org/name/resolve/main/x": true,
		"http://huggingface.co/org/name/resolve/main/x":  false, // token would go in cleartext
		"https://evil.example/x":                         false,
		"http://10.0.0.1:9000/admin":                     false,
	} {
		if err := im.CheckSource(u); (err == nil) != ok {
			t.Errorf("%s: allowed=%v, want %v (%v)", u, err == nil, ok, err)
		}
	}
}
