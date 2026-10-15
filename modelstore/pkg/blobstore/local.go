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
	"os"
	"path/filepath"

	"k8s.io/klog/v2"
)

// Local keeps blobs as files under dir/blobs/<sha>.
type Local struct {
	dir string
}

// NewLocal opens (creating if needed) a directory-backed store.
func NewLocal(dir string) (*Local, error) {
	if err := os.MkdirAll(filepath.Join(dir, "blobs"), 0o755); err != nil {
		return nil, fmt.Errorf("creating blob directory: %w", err)
	}
	return &Local{dir: dir}, nil
}

func (l *Local) path(sha string) string { return filepath.Join(l.dir, "blobs", sha) }

func (l *Local) Stat(ctx context.Context, sha string) (int64, error) {
	info, err := os.Stat(l.path(sha))
	if errors.Is(err, os.ErrNotExist) {
		return 0, ErrNotFound
	}
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

func (l *Local) Put(ctx context.Context, sha string, r io.Reader) (string, int64, error) {
	tmp, err := os.CreateTemp(filepath.Join(l.dir, "blobs"), "upload-*")
	if err != nil {
		return "", 0, fmt.Errorf("creating temp file: %w", err)
	}
	tmpPath := tmp.Name()
	published := false
	log := klog.FromContext(ctx)
	defer func() {
		if err := tmp.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			log.Error(err, "closing temp file", "path", tmpPath)
		}
		if !published {
			if err := os.Remove(tmpPath); err != nil && !errors.Is(err, os.ErrNotExist) {
				log.Error(err, "removing temp file", "path", tmpPath)
			}
		}
	}()
	dw := newDigestWriter(tmp)
	if err := dw.copyFrom(r); err != nil {
		return "", dw.n, fmt.Errorf("writing blob: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", dw.n, fmt.Errorf("closing temp file: %w", err)
	}
	got, err := dw.check(sha)
	if err != nil {
		return got, dw.n, err
	}
	if err := os.Rename(tmpPath, l.path(got)); err != nil {
		return got, dw.n, fmt.Errorf("publishing blob: %w", err)
	}
	published = true
	return got, dw.n, nil
}

func (l *Local) Open(ctx context.Context, sha string, offset, length int64) (io.ReadCloser, error) {
	f, err := os.Open(l.path(sha))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if offset > 0 {
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			if cerr := f.Close(); cerr != nil {
				klog.FromContext(ctx).Error(cerr, "closing blob after failed seek", "sha", sha)
			}
			return nil, err
		}
	}
	if length < 0 {
		return f, nil
	}
	return &limitedFile{Reader: io.LimitReader(f, length), f: f}, nil
}

type limitedFile struct {
	io.Reader
	f *os.File
}

func (l *limitedFile) Close() error { return l.f.Close() }

func (l *Local) DirectURL(sha string) string { return "" }

func (l *Local) String() string { return "local:" + l.dir }
