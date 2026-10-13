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
	"context"
	"errors"
	"fmt"
	"io"
	"path"
	"strings"

	"cloud.google.com/go/storage"
	"github.com/google/uuid"
	"k8s.io/klog/v2"
)

// GCS keeps blobs as objects <prefix>/_blobs/<sha> in a bucket.
//
// Put streams into a temporary object under <prefix>/_tmp/, verifying
// the sha256 as the bytes pass, then copies it server-side to its
// final name and deletes the temporary: a reader listing or opening
// _blobs never sees a partial object, and a mismatch publishes nothing.
type GCS struct {
	client *storage.Client
	bucket string
	prefix string
}

// NewGCS opens gs://bucket[/prefix] with application default credentials
// (workload identity on GKE).
func NewGCS(ctx context.Context, location string) (*GCS, error) {
	rest := strings.TrimPrefix(location, "gs://")
	bucket, prefix, _ := strings.Cut(rest, "/")
	if bucket == "" {
		return nil, fmt.Errorf("invalid GCS location %q: want gs://bucket[/prefix]", location)
	}
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating GCS client: %w", err)
	}
	return &GCS{client: client, bucket: bucket, prefix: strings.Trim(prefix, "/")}, nil
}

func (g *GCS) object(sha string) string { return path.Join(g.prefix, "_blobs", sha) }

func (g *GCS) Stat(ctx context.Context, sha string) (int64, error) {
	attrs, err := g.client.Bucket(g.bucket).Object(g.object(sha)).Attrs(ctx)
	if errors.Is(err, storage.ErrObjectNotExist) {
		return 0, ErrNotFound
	}
	if err != nil {
		return 0, err
	}
	return attrs.Size, nil
}

func (g *GCS) Put(ctx context.Context, sha string, r io.Reader) (string, int64, error) {
	bkt := g.client.Bucket(g.bucket)
	tmpName := path.Join(g.prefix, "_tmp", uuid.NewString())
	tmp := bkt.Object(tmpName)
	w := tmp.NewWriter(ctx)
	w.ChunkSize = 64 << 20 // fewer resumable-upload round trips for multi-GB files
	// discard removes the temporary object after a failure; the
	// original error is what the caller sees.
	discard := func() {
		if err := tmp.Delete(context.WithoutCancel(ctx)); err != nil && !errors.Is(err, storage.ErrObjectNotExist) {
			klog.FromContext(ctx).Error(err, "removing temporary object", "object", tmpName)
		}
	}
	dw := newDigestWriter(w)
	if _, err := io.Copy(dw, r); err != nil {
		if cerr := w.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing failed upload", "object", tmpName, "error", cerr.Error())
		}
		discard()
		return "", dw.n, fmt.Errorf("uploading blob: %w", err)
	}
	if err := w.Close(); err != nil {
		discard()
		return "", dw.n, fmt.Errorf("finishing upload: %w", err)
	}
	got, err := dw.check(sha)
	if err != nil {
		discard()
		return got, dw.n, err
	}
	// Publish: server-side copy to the final name (no bytes move through
	// here), then drop the temporary.
	copier := bkt.Object(g.object(got)).CopierFrom(tmp)
	copier.Metadata = map[string]string{"sha256": got}
	if _, err := copier.Run(ctx); err != nil {
		discard()
		return got, dw.n, fmt.Errorf("publishing blob: %w", err)
	}
	if err := tmp.Delete(ctx); err != nil && !errors.Is(err, storage.ErrObjectNotExist) {
		return got, dw.n, fmt.Errorf("removing temporary object: %w", err)
	}
	return got, dw.n, nil
}

func (g *GCS) Open(ctx context.Context, sha string, offset, length int64) (io.ReadCloser, error) {
	rc, err := g.client.Bucket(g.bucket).Object(g.object(sha)).NewRangeReader(ctx, offset, length)
	if errors.Is(err, storage.ErrObjectNotExist) {
		return nil, ErrNotFound
	}
	return rc, err
}

func (g *GCS) DirectURL(sha string) string {
	return "gs://" + g.bucket + "/" + g.object(sha)
}

func (g *GCS) String() string { return "gs://" + g.bucket + "/" + g.prefix }
