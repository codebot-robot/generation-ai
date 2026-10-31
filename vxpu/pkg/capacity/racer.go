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

package capacity

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

// IsNotFound reports whether the error represents a resource not found.
func IsNotFound(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrNotFound) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "not found") || strings.Contains(msg, "notfound") || strings.Contains(msg, "404")
}

// IsUnsupportedZone reports whether the error indicates that a zone does not offer the requested type.
func IsUnsupportedZone(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrUnsupportedZone) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unsupported") ||
		strings.Contains(msg, "not available in zone") ||
		strings.Contains(msg, "not available in this zone") ||
		strings.Contains(msg, "machine type not found") ||
		strings.Contains(msg, "accelerator type not found") ||
		strings.Contains(msg, "invalid_usage")
}

// DeterministicName generates a consistent instance name for a given zone.
func DeterministicName(baseName, zone string) string {
	if baseName == "" {
		baseName = "vxpu-race"
	}
	baseName = strings.TrimSuffix(baseName, "-")
	return fmt.Sprintf("%s-%s", baseName, zone)
}

// zoneTarget tracks the running state for one zone in the race.
type zoneTarget struct {
	zone      string
	name      string
	spec      MachineSpec
	reqTime   time.Time
	lastState InstanceStatus
	created   bool
}

// Racer coordinates flex-start requests across multiple zones.
type Racer struct {
	backend Backend
	opts    RaceOptions
}

// NewRacer constructs a Racer with the given backend and options.
func NewRacer(backend Backend, opts RaceOptions) *Racer {
	if opts.PollInterval <= 0 {
		opts.PollInterval = 2 * time.Second
	}
	return &Racer{
		backend: backend,
		opts:    opts,
	}
}

// Race executes a capacity race across the configured zones.
func Race(ctx context.Context, backend Backend, opts RaceOptions) (*Winner, error) {
	racer := NewRacer(backend, opts)
	return racer.Run(ctx)
}

// Run starts the race and blocks until a request is granted, the context is cancelled, or the deadline expires.
func (r *Racer) Run(ctx context.Context) (*Winner, error) {
	log := klog.FromContext(ctx)
	if len(r.opts.Zones) == 0 {
		return nil, errors.New("capacity race requires at least one zone")
	}

	raceCtx := ctx
	var cancel context.CancelFunc
	if r.opts.Timeout > 0 {
		raceCtx, cancel = context.WithTimeout(ctx, r.opts.Timeout)
		defer cancel()
	}

	baseName := r.opts.Spec.Name
	if baseName == "" {
		baseName = "vxpu-race"
	}

	targets := make(map[string]*zoneTarget, len(r.opts.Zones))
	for _, z := range r.opts.Zones {
		name := DeterministicName(baseName, z)
		spec := r.opts.Spec
		spec.Name = name
		targets[z] = &zoneTarget{
			zone:    z,
			name:    name,
			spec:    spec,
			reqTime: time.Now(),
		}
	}

	// createdTargets tracks all instances created or adopted that must be cleaned up on abort.
	var targetsMu sync.Mutex
	createdTargets := make(map[string]*zoneTarget)

	markCreated := func(t *zoneTarget) {
		targetsMu.Lock()
		defer targetsMu.Unlock()
		t.created = true
		createdTargets[t.zone] = t
	}

	// Initialize requests across all zones.
	var initWg sync.WaitGroup
	var earlyWinner *Winner
	var earlyWinnerMu sync.Mutex

	for _, t := range targets {
		initWg.Add(1)
		go func(tgt *zoneTarget) {
			defer initWg.Done()
			inst, err := r.backend.Describe(raceCtx, tgt.zone, tgt.name)
			if err == nil {
				// The instance already exists. Adopt it.
				markCreated(tgt)
				tgt.lastState = inst.Status
				log.Info("Adopted existing flex-start request", "zone", tgt.zone, "name", tgt.name, "status", string(inst.Status), "timestamp", time.Now().UTC().Format(time.RFC3339))
				if inst.Status == StatusRunning {
					earlyWinnerMu.Lock()
					if earlyWinner == nil {
						earlyWinner = &Winner{
							Name: tgt.name,
							Zone: tgt.zone,
							Wait: time.Since(tgt.reqTime),
						}
					}
					earlyWinnerMu.Unlock()
					return
				}
				if inst.Status == StatusTerminated {
					// Instance terminated earlier. Clean it up and recreate it.
					log.Info("Adopted instance was terminated; recreating", "zone", tgt.zone, "name", tgt.name, "timestamp", time.Now().UTC().Format(time.RFC3339))
					if err := r.backend.Delete(context.WithoutCancel(raceCtx), tgt.zone, tgt.name); err != nil && !IsNotFound(err) {
						log.Error(err, "Failed to delete terminated adopted instance", "zone", tgt.zone, "name", tgt.name)
					}
					r.waitForDeletion(context.WithoutCancel(raceCtx), tgt.zone, tgt.name)
					tgt.reqTime = time.Now()
					if cerr := r.backend.Create(raceCtx, tgt.zone, tgt.spec); cerr != nil {
						log.Error(cerr, "Failed to recreate expired instance", "zone", tgt.zone, "name", tgt.name)
					} else {
						tgt.lastState = StatusPending
						log.Info("Requested flex-start instance", "zone", tgt.zone, "name", tgt.name, "state", "requested", "timestamp", time.Now().UTC().Format(time.RFC3339))
					}
				}
				return
			}

			if !IsNotFound(err) {
				log.Error(err, "Failed to describe instance before creation", "zone", tgt.zone, "name", tgt.name)
			}

			// Instance does not exist. Create it.
			tgt.reqTime = time.Now()
			cerr := r.backend.Create(raceCtx, tgt.zone, tgt.spec)
			if cerr != nil {
				if IsUnsupportedZone(cerr) {
					log.Info("Zone does not offer machine type; dropping zone from race", "zone", tgt.zone, "error", cerr.Error(), "timestamp", time.Now().UTC().Format(time.RFC3339))
					return
				}
				log.Error(cerr, "Failed to create flex-start request", "zone", tgt.zone, "name", tgt.name)
				return
			}
			markCreated(tgt)
			tgt.lastState = StatusPending
			log.Info("Requested flex-start instance", "zone", tgt.zone, "name", tgt.name, "state", "requested", "timestamp", time.Now().UTC().Format(time.RFC3339))
		}(t)
	}
	initWg.Wait()

	// If an adopted request was already RUNNING, finish immediately.
	if earlyWinner != nil {
		log.Info("Flex-start request granted", "zone", earlyWinner.Zone, "name", earlyWinner.Name, "state", "granted", "waited", earlyWinner.Wait, "timestamp", time.Now().UTC().Format(time.RFC3339))
		r.cancelOthers(ctx, createdTargets, earlyWinner.Zone)
		log.Info("Total wait", "winner", earlyWinner.Name, "zone", earlyWinner.Zone, "total_wait", earlyWinner.Wait, "timestamp", time.Now().UTC().Format(time.RFC3339))
		return earlyWinner, nil
	}

	targetsMu.Lock()
	activeCount := len(createdTargets)
	targetsMu.Unlock()
	if activeCount == 0 {
		return nil, errors.New("no active flex-start requests could be started in the given zones")
	}

	// Poll loop until a winner is found or context terminates.
	ticker := time.NewTicker(r.opts.PollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-raceCtx.Done():
			// Context cancelled or timed out.
			cleanupCtx := context.WithoutCancel(ctx)
			r.cancelAll(cleanupCtx, createdTargets)
			return nil, raceCtx.Err()
		case <-ticker.C:
		}

		winner, err := r.pollActiveTargets(raceCtx, createdTargets, markCreated)
		if err != nil {
			cleanupCtx := context.WithoutCancel(ctx)
			r.cancelAll(cleanupCtx, createdTargets)
			return nil, err
		}
		if winner != nil {
			log.Info("Flex-start request granted", "zone", winner.Zone, "name", winner.Name, "state", "granted", "waited", winner.Wait, "timestamp", time.Now().UTC().Format(time.RFC3339))
			// Winner found! Cancel all other requests immediately in parallel.
			r.cancelOthers(ctx, createdTargets, winner.Zone)
			log.Info("Total wait", "winner", winner.Name, "zone", winner.Zone, "total_wait", winner.Wait, "timestamp", time.Now().UTC().Format(time.RFC3339))
			return winner, nil
		}
	}
}

// pollActiveTargets checks the status of all active requests in parallel.
func (r *Racer) pollActiveTargets(ctx context.Context, targets map[string]*zoneTarget, markCreated func(*zoneTarget)) (*Winner, error) {
	log := klog.FromContext(ctx)
	var wg sync.WaitGroup
	var winner *Winner
	var winnerMu sync.Mutex

	for _, tgt := range targets {
		wg.Add(1)
		go func(t *zoneTarget) {
			defer wg.Done()
			inst, err := r.backend.Describe(ctx, t.zone, t.name)
			if err != nil {
				if IsNotFound(err) {
					// Instance disappeared or expired. Recreate it.
					log.Info("Flex-start request not found; recreating", "zone", t.zone, "name", t.name, "timestamp", time.Now().UTC().Format(time.RFC3339))
					t.reqTime = time.Now()
					if cerr := r.backend.Create(ctx, t.zone, t.spec); cerr != nil {
						log.Error(cerr, "Failed to recreate not-found instance", "zone", t.zone, "name", t.name)
					} else {
						markCreated(t)
						t.lastState = StatusPending
						log.Info("Requested flex-start instance", "zone", t.zone, "name", t.name, "state", "requested", "timestamp", time.Now().UTC().Format(time.RFC3339))
					}
					return
				}
				// Transient describe error; retry next tick.
				return
			}

			if inst.Status == StatusRunning {
				winnerMu.Lock()
				if winner == nil {
					winner = &Winner{
						Name: t.name,
						Zone: t.zone,
						Wait: time.Since(t.reqTime),
					}
				}
				winnerMu.Unlock()
				return
			}

			if inst.Status == StatusTerminated {
				// Request expired or terminated without capacity.
				log.Info("Flex-start request expired; recreating", "zone", t.zone, "name", t.name, "reason", inst.StatusMessage, "timestamp", time.Now().UTC().Format(time.RFC3339))
				if err := r.backend.Delete(context.WithoutCancel(ctx), t.zone, t.name); err != nil && !IsNotFound(err) {
					log.Error(err, "Failed to delete expired instance", "zone", t.zone, "name", t.name)
				}
				r.waitForDeletion(context.WithoutCancel(ctx), t.zone, t.name)
				t.reqTime = time.Now()
				if cerr := r.backend.Create(ctx, t.zone, t.spec); cerr != nil {
					log.Error(cerr, "Failed to recreate expired instance", "zone", t.zone, "name", t.name)
				} else {
					markCreated(t)
					t.lastState = StatusPending
					log.Info("Requested flex-start instance", "zone", t.zone, "name", t.name, "state", "requested", "timestamp", time.Now().UTC().Format(time.RFC3339))
				}
				return
			}

			if inst.Status == StatusPending && t.lastState != StatusPending {
				t.lastState = StatusPending
				log.Info("Flex-start request pending", "zone", t.zone, "name", t.name, "state", "pending", "waited", time.Since(t.reqTime).Round(time.Second), "timestamp", time.Now().UTC().Format(time.RFC3339))
			}
		}(tgt)
	}

	wg.Wait()
	return winner, nil
}

// cancelOthers deletes all active requests except the winner, in parallel, and verifies they are gone.
func (r *Racer) cancelOthers(ctx context.Context, targets map[string]*zoneTarget, winnerZone string) {
	cleanupCtx := context.WithoutCancel(ctx)
	var wg sync.WaitGroup
	for zone, tgt := range targets {
		if zone == winnerZone {
			continue
		}
		wg.Add(1)
		go func(t *zoneTarget) {
			defer wg.Done()
			r.deleteAndConfirm(cleanupCtx, t.zone, t.name)
		}(tgt)
	}
	wg.Wait()
}

// cancelAll deletes all created requests in parallel, and verifies they are gone.
func (r *Racer) cancelAll(ctx context.Context, targets map[string]*zoneTarget) {
	var wg sync.WaitGroup
	for _, tgt := range targets {
		wg.Add(1)
		go func(t *zoneTarget) {
			defer wg.Done()
			r.deleteAndConfirm(ctx, t.zone, t.name)
		}(tgt)
	}
	wg.Wait()
}

// deleteAndConfirm calls Delete and retries until the instance is confirmed deleted.
func (r *Racer) deleteAndConfirm(ctx context.Context, zone, name string) {
	log := klog.FromContext(ctx)
	for {
		delCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		err := r.backend.Delete(delCtx, zone, name)
		cancel()
		if err != nil {
			if IsNotFound(err) {
				log.Info("Flex-start request deleted", "zone", zone, "name", name, "state", "deleted", "timestamp", time.Now().UTC().Format(time.RFC3339))
				return
			}
			// Transient delete error; retry after short backoff.
			select {
			case <-ctx.Done():
				return
			case <-time.After(300 * time.Millisecond):
				continue
			}
		}

		// Delete succeeded or was queued. Now confirm the instance is gone.
		if r.waitForDeletion(ctx, zone, name) {
			log.Info("Flex-start request deleted", "zone", zone, "name", name, "state", "deleted", "timestamp", time.Now().UTC().Format(time.RFC3339))
			return
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(300 * time.Millisecond):
		}
	}
}

// waitForDeletion polls Describe until the instance is not found.
// If the instance unexpectedly turns RUNNING (a late grant), it issues another Delete.
func (r *Racer) waitForDeletion(ctx context.Context, zone, name string) bool {
	for i := 0; i < 60; i++ {
		descCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		inst, err := r.backend.Describe(descCtx, zone, name)
		cancel()
		if err != nil && IsNotFound(err) {
			return true
		}
		if err == nil && inst.Status == StatusRunning {
			// A late grant was filled before deletion finished. Delete it again immediately.
			delCtx, delCancel := context.WithTimeout(ctx, 10*time.Second)
			if derr := r.backend.Delete(delCtx, zone, name); derr != nil && !IsNotFound(derr) {
				log := klog.FromContext(ctx)
				log.Error(derr, "Failed to delete late-granted instance", "zone", zone, "name", name)
			}
			delCancel()
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(200 * time.Millisecond):
		}
	}
	return false
}
