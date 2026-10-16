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
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
	"k8s.io/klog/v2"
)

// URLSigner generates a pre-signed URL for an object in a bucket.
type URLSigner interface {
	SignURL(ctx context.Context, bucket, object string, expires time.Time) (string, error)
}

// URLSignerFunc adapts a function to URLSigner.
type URLSignerFunc func(ctx context.Context, bucket, object string, expires time.Time) (string, error)

func (f URLSignerFunc) SignURL(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
	return f(ctx, bucket, object, expires)
}

type cachedURL struct {
	url       string
	expiresAt time.Time
	validity  time.Duration
}

type negativeEntry struct {
	retryAfter time.Time
	backoff    time.Duration
	err        error
}

// URLSignerCache caches pre-signed URLs in-process per blob and refreshes
// them when less than half their validity remains.
type URLSignerCache struct {
	bucket   string
	signer   URLSigner
	validity time.Duration
	now      func() time.Time

	mu        sync.RWMutex
	blobs     map[string]cachedURL
	negatives map[string]negativeEntry
	flight    singleflight.Group
}

// DefaultValidity is 7 days, the maximum validity for GCS V4 signed URLs.
const DefaultValidity = 7 * 24 * time.Hour

// Initial negative TTL and max backoff for failed signing attempts.
const (
	initialNegativeTTL = 5 * time.Second
	maxNegativeTTL     = 5 * time.Minute
)

// NewURLSignerCache creates an in-process signed URL cache.
func NewURLSignerCache(bucket string, signer URLSigner, validity time.Duration, now func() time.Time) *URLSignerCache {
	if validity <= 0 {
		validity = DefaultValidity
	}
	if now == nil {
		now = time.Now
	}
	return &URLSignerCache{
		bucket:    bucket,
		signer:    signer,
		validity:  validity,
		now:       now,
		blobs:     make(map[string]cachedURL),
		negatives: make(map[string]negativeEntry),
	}
}

// Get returns a cached signed URL for object, or signs a new one if not
// cached or if less than half its validity remains.
func (c *URLSignerCache) Get(ctx context.Context, sha, object string) (string, error) {
	if c == nil || c.signer == nil {
		return "", nil
	}

	if err := ctx.Err(); err != nil {
		return "", err
	}

	now := c.now()

	c.mu.RLock()
	entry, ok := c.blobs[sha]
	neg, hasNeg := c.negatives[sha]
	c.mu.RUnlock()

	if ok {
		remaining := entry.expiresAt.Sub(now)
		if remaining >= entry.validity/2 {
			return entry.url, nil
		}
	}

	if hasNeg && now.Before(neg.retryAfter) {
		if ok && entry.expiresAt.After(now) {
			return entry.url, nil
		}
		return "", neg.err
	}

	ch := c.flight.DoChan(sha, func() (any, error) {
		now := c.now()
		c.mu.RLock()
		entry, ok := c.blobs[sha]
		neg, hasNeg := c.negatives[sha]
		c.mu.RUnlock()

		if ok {
			remaining := entry.expiresAt.Sub(now)
			if remaining >= entry.validity/2 {
				return entry.url, nil
			}
		}

		if hasNeg && now.Before(neg.retryAfter) {
			if ok && entry.expiresAt.After(now) {
				return entry.url, nil
			}
			return "", neg.err
		}

		expires := now.Add(c.validity)
		signCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		url, err := c.signer.SignURL(signCtx, c.bucket, object, expires)
		if err != nil {
			if ok && entry.expiresAt.After(now) {
				klog.FromContext(ctx).Error(err, "refreshing direct URL failed; using unexpired cached URL", "sha", sha)
				return entry.url, nil
			}

			if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				c.mu.Lock()
				backoff := initialNegativeTTL
				if hasNeg && neg.backoff > 0 {
					backoff = neg.backoff * 2
					if backoff > maxNegativeTTL {
						backoff = maxNegativeTTL
					}
				}
				c.negatives[sha] = negativeEntry{
					retryAfter: now.Add(backoff),
					backoff:    backoff,
					err:        err,
				}
				c.mu.Unlock()
			}
			return "", err
		}

		c.mu.Lock()
		delete(c.negatives, sha)
		c.blobs[sha] = cachedURL{
			url:       url,
			expiresAt: expires,
			validity:  c.validity,
		}
		c.mu.Unlock()

		return url, nil
	})

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case res := <-ch:
		if res.Err != nil {
			return "", res.Err
		}
		return res.Val.(string), nil
	}
}
