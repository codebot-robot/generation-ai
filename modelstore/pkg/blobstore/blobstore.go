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

// Package blobstore stores model files by their sha256.
//
// A blob is immutable and named by the sha256 of its bytes, so a store
// never needs to update in place: Put verifies the digest while the
// bytes stream through and only then publishes the blob under its
// name, and readers never observe a partial blob. Two backends: a
// local directory (kind, tests) and a Google Cloud Storage bucket (the
// cluster's weight store; readers with their own bucket access can
// range-read the object directly, see Store.DirectURL).
package blobstore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ErrNotFound is returned by Stat and Open for a blob the store does not hold.
var ErrNotFound = errors.New("blob not found")

// Store is a content-addressed blob store.
type Store interface {
	// Stat reports the size of a blob, or ErrNotFound.
	Stat(ctx context.Context, sha string) (int64, error)
	// Put stores the bytes of r as the blob sha, verifying the digest;
	// with sha == "" the blob is named by its content. Returns the
	// blob's sha and the bytes written. A mismatch leaves no blob.
	Put(ctx context.Context, sha string, r io.Reader) (string, int64, error)
	// Open reads length bytes of the blob starting at offset; length < 0
	// reads to the end.
	Open(ctx context.Context, sha string, offset, length int64) (io.ReadCloser, error)
	// DirectURL is a URL a reader with its own access to the backing
	// store can use to read the blob without going through this
	// server (a gs:// URL for the GCS backend); "" when there is none.
	DirectURL(sha string) string
	// String describes the backend for logs.
	String() string
}

// Open opens a store from its location: a gs://bucket/prefix URL or a
// local directory path.
func Open(ctx context.Context, location string) (Store, error) {
	if strings.HasPrefix(location, "gs://") {
		return NewGCS(ctx, location)
	}
	return NewLocal(location)
}

// ValidSHA reports whether s is a lowercase hex sha256 digest.
func ValidSHA(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// digestWriter hashes what passes through it.
type digestWriter struct {
	w io.Writer
	h hashWriter
	n int64
}

type hashWriter interface {
	io.Writer
	Sum(b []byte) []byte
}

func newDigestWriter(w io.Writer) *digestWriter {
	return &digestWriter{w: w, h: sha256.New()}
}

func (d *digestWriter) Write(p []byte) (int, error) {
	n, err := d.w.Write(p)
	d.n += int64(n)
	if _, herr := d.h.Write(p[:n]); herr != nil {
		return n, herr
	}
	return n, err
}

func (d *digestWriter) sum() string {
	return hex.EncodeToString(d.h.Sum(nil))
}

// check verifies the digest against sha; an empty sha accepts any and
// returns the computed one.
func (d *digestWriter) check(sha string) (string, error) {
	got := d.sum()
	if sha != "" && got != sha {
		return got, fmt.Errorf("sha256 mismatch: expected %s, got %s", sha, got)
	}
	return got, nil
}
