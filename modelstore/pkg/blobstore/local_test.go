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

package blobstore

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// Refs are flat files, as in the GCS backend: a tag of "org" named
// "name" and the tags of "org/name" coexist, a tag may end in ".tmp",
// and a stale temp file from a killed process is swept at open.
func TestLocalRefsAreFlat(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "blobs", "upload-stale"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	l, err := NewLocal(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "blobs", "upload-stale")); !os.IsNotExist(err) {
		t.Errorf("stale upload temp not swept: %v", err)
	}
	ctx := t.Context()
	for key, v := range map[string]string{"tags/org/name": "a", "tags/org/name/main": "b", "tags/org/name/v1.tmp": "c"} {
		if err := l.PutRef(ctx, key, []byte(v)); err != nil {
			t.Fatalf("PutRef %s: %v", key, err)
		}
	}
	if b, err := l.GetRef(ctx, "tags/org/name"); err != nil || string(b) != "a" {
		t.Errorf("tag of org: %q %v", b, err)
	}
	got, err := l.ListRefs(ctx, "tags/org/name/")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"tags/org/name/main", "tags/org/name/v1.tmp"}; !reflect.DeepEqual(got, want) {
		t.Errorf("ListRefs: got %v want %v", got, want)
	}
	if got, _ := l.ListRefs(ctx, "tags/org/"); len(got) != 3 {
		t.Errorf("ListRefs tags/org/: %v", got)
	}
}
