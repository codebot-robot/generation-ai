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
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestURLSignerCache_InitialSignAndCacheHit(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d&exp=%d", bucket, object, count, expires.Unix()), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)

	sha := "1111111111111111111111111111111111111111111111111111111111111111"
	obj := "_blobs/" + sha

	// Initial call signs and caches
	u1, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}

	// 1 day later (6 days remaining >= 3.5 days): cache hit, no new sign call
	nowTime = nowTime.Add(24 * time.Hour)
	u2, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u1 != u2 {
		t.Fatalf("expected same URL, got %s vs %s", u1, u2)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}

	// 3 days from start (4 days remaining >= 3.5 days): still cache hit
	nowTime = nowTime.Add(48 * time.Hour)
	u3, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u1 != u3 {
		t.Fatalf("expected same URL, got %s vs %s", u1, u3)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}
}

func TestURLSignerCache_RefreshWhenLessThanHalfValidityRemains(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d&exp=%d", bucket, object, count, expires.Unix()), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "2222222222222222222222222222222222222222222222222222222222222222"
	obj := "_blobs/" + sha

	u1, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if signCalls.Load() != 1 {
		t.Fatalf("expected 1 call, got %d", signCalls.Load())
	}

	// Advance time to 3.6 days later (3.4 days remaining < 3.5 days -> triggers refresh!)
	nowTime = nowTime.Add(3*24*time.Hour + 14*time.Hour + 24*time.Minute)
	u2, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if signCalls.Load() != 2 {
		t.Fatalf("expected refresh to trigger sign call 2, got %d", signCalls.Load())
	}
	if u1 == u2 {
		t.Fatalf("expected refreshed URL to differ from initial URL")
	}

	// Immediately after refresh (remaining ~7 days >= 3.5 days): cache hit with new URL
	nowTime = nowTime.Add(time.Hour)
	u3, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u2 != u3 {
		t.Fatalf("expected same refreshed URL, got %s vs %s", u2, u3)
	}
	if signCalls.Load() != 2 {
		t.Fatalf("expected still 2 calls, got %d", signCalls.Load())
	}
}

func TestURLSignerCache_ExpiredURL(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d", bucket, object, count), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "3333333333333333333333333333333333333333333333333333333333333333"
	obj := "_blobs/" + sha

	_, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatal(err)
	}

	// Advance time past the 7 day expiration (e.g. 10 days later)
	nowTime = nowTime.Add(10 * 24 * time.Hour)
	_, err = cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatal(err)
	}
	if signCalls.Load() != 2 {
		t.Fatalf("expected 2 calls, got %d", signCalls.Load())
	}
}

func TestURLSignerCache_ErrorFallbackToUnexpired(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	failSign := atomic.Bool{}

	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		signCalls.Add(1)
		if failSign.Load() {
			return "", errors.New("simulated IAM signing failure")
		}
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s", bucket, object), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "4444444444444444444444444444444444444444444444444444444444444444"
	obj := "_blobs/" + sha

	u1, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatal(err)
	}

	// Advance time to 4 days later (remaining 3 days < 3.5 days, but not expired).
	// Enable failure.
	nowTime = nowTime.Add(4 * 24 * time.Hour)
	failSign.Store(true)

	// Cache should fallback to the still-valid cached URL rather than failing
	u2, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("expected fallback to unexpired cached URL, got error: %v", err)
	}
	if u1 != u2 {
		t.Fatalf("expected u1 fallback, got %s", u2)
	}

	// Advance time past 7 days (now fully expired)
	nowTime = nowTime.Add(4 * 24 * time.Hour)
	_, err = cache.Get(ctx, sha, obj)
	if err == nil {
		t.Fatalf("expected error when cached URL is expired and signer fails")
	}
}

func TestURLSignerCache_ConcurrentRequestsJoinOneSign(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		time.Sleep(20 * time.Millisecond) // simulate IAM network latency
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d", bucket, object, count), nil
	})

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nil)
	sha := "5555555555555555555555555555555555555555555555555555555555555555"
	obj := "_blobs/" + sha

	const concurrency = 20
	var wg sync.WaitGroup
	urls := make([]string, concurrency)
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			u, err := cache.Get(ctx, sha, obj)
			if err != nil {
				t.Errorf("goroutine %d: %v", idx, err)
				return
			}
			urls[idx] = u
		}(i)
	}
	wg.Wait()

	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected exactly 1 sign call across %d concurrent requests, got %d", concurrency, calls)
	}
	for i, u := range urls {
		if u != urls[0] {
			t.Errorf("url[%d] (%s) != url[0] (%s)", i, u, urls[0])
		}
	}
}

func TestURLSignerCache_NegativeCacheWithBackoff(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	failSign := atomic.Bool{}
	failSign.Store(true)

	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		signCalls.Add(1)
		if failSign.Load() {
			return "", errors.New("simulated IAM failure")
		}
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s", bucket, object), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "6666666666666666666666666666666666666666666666666666666666666666"
	obj := "_blobs/" + sha

	// First attempt fails
	_, err := cache.Get(ctx, sha, obj)
	if err == nil {
		t.Fatal("expected error on first sign attempt")
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 call, got %d", calls)
	}

	// Immediate retry within initial backoff (5s) hits negative cache without calling SignURL
	nowTime = nowTime.Add(2 * time.Second)
	_, err = cache.Get(ctx, sha, obj)
	if err == nil {
		t.Fatal("expected error from negative cache")
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected still 1 call (negative cache hit), got %d", calls)
	}

	// Advance time past initial backoff (e.g. 6s from start) -> retries SignURL
	nowTime = nowTime.Add(4 * time.Second) // total 6s
	_, err = cache.Get(ctx, sha, obj)
	if err == nil {
		t.Fatal("expected error on second attempt")
	}
	if calls := signCalls.Load(); calls != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}

	// Next backoff doubled to 10s. Advance 7s (total 13s) -> still in backoff
	nowTime = nowTime.Add(7 * time.Second)
	_, err = cache.Get(ctx, sha, obj)
	if err == nil {
		t.Fatal("expected error from negative cache")
	}
	if calls := signCalls.Load(); calls != 2 {
		t.Fatalf("expected still 2 calls (negative cache hit), got %d", calls)
	}

	// Advance past 10s backoff (advance 4s more -> 11s after second failure)
	nowTime = nowTime.Add(4 * time.Second)
	failSign.Store(false) // recovery!
	u, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("expected success after backoff, got %v", err)
	}
	if calls := signCalls.Load(); calls != 3 {
		t.Fatalf("expected 3 calls, got %d", calls)
	}
	if u == "" {
		t.Fatal("expected non-empty URL")
	}
}

func TestURLSignerCache_ContextCancellationDoesNotPoisonCache(t *testing.T) {
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s", bucket, object), nil
	})

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nil)
	sha := "7777777777777777777777777777777777777777777777777777777777777777"
	obj := "_blobs/" + sha

	// Cancelled context before call
	canceledCtx, cancel := context.WithCancel(t.Context())
	cancel()

	_, err := cache.Get(canceledCtx, sha, obj)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if calls := signCalls.Load(); calls != 0 {
		t.Fatalf("expected 0 sign calls, got %d", calls)
	}

	// A normal request should succeed immediately and not be blocked by negative cache
	u, err := cache.Get(t.Context(), sha, obj)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}
	if u == "" {
		t.Fatal("expected non-empty URL")
	}
}

func TestURLSignerCache_FirstCallerCancellationLeavesSignAlive(t *testing.T) {
	signStarted := make(chan struct{})
	blockSign := make(chan struct{})

	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		close(signStarted)
		select {
		case <-blockSign:
			return fmt.Sprintf("https://storage.googleapis.com/%s/%s", bucket, object), nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nil)
	sha := "8888888888888888888888888888888888888888888888888888888888888888"
	obj := "_blobs/" + sha

	ctx1, cancel1 := context.WithCancel(t.Context())

	errCh1 := make(chan error, 1)
	go func() {
		_, err := cache.Get(ctx1, sha, obj)
		errCh1 <- err
	}()

	<-signStarted
	// Cancel caller 1's context
	cancel1()

	err1 := <-errCh1
	if !errors.Is(err1, context.Canceled) {
		t.Fatalf("caller 1 expected context.Canceled, got %v", err1)
	}

	// Unblock signer
	close(blockSign)

	// Caller 2 should succeed with valid cached URL
	u2, err := cache.Get(t.Context(), sha, obj)
	if err != nil {
		t.Fatalf("caller 2 expected success, got %v", err)
	}
	if u2 == "" {
		t.Fatal("caller 2 expected non-empty URL")
	}
}
