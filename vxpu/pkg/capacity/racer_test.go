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
	"sync"
	"testing"
	"time"
)

type fakeBackend struct {
	mu         sync.Mutex
	instances  map[string]*Instance
	creates    []string
	deletes    []string
	describes  []string
	failCreate map[string]error
	failDelete map[string]error
	onDescribe func(zone, name string, inst *Instance)
	onDelete   func(zone, name string)
}

func newFakeBackend() *fakeBackend {
	return &fakeBackend{
		instances:  make(map[string]*Instance),
		failCreate: make(map[string]error),
		failDelete: make(map[string]error),
	}
}

func (f *fakeBackend) key(zone, name string) string {
	return fmt.Sprintf("%s/%s", zone, name)
}

func (f *fakeBackend) Create(ctx context.Context, zone string, spec MachineSpec) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.creates = append(f.creates, zone)
	if err := f.failCreate[zone]; err != nil {
		return err
	}
	k := f.key(zone, spec.Name)
	f.instances[k] = &Instance{
		Name:   spec.Name,
		Zone:   zone,
		Status: StatusPending,
	}
	return nil
}

func (f *fakeBackend) Describe(ctx context.Context, zone string, name string) (*Instance, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.describes = append(f.describes, zone)
	k := f.key(zone, name)
	inst, ok := f.instances[k]
	if !ok || inst == nil {
		return nil, ErrNotFound
	}
	if f.onDescribe != nil {
		f.onDescribe(zone, name, inst)
	}
	copy := *inst
	return &copy, nil
}

func (f *fakeBackend) Delete(ctx context.Context, zone string, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deletes = append(f.deletes, zone)
	if err := f.failDelete[zone]; err != nil {
		return err
	}
	k := f.key(zone, name)
	if _, ok := f.instances[k]; !ok {
		return ErrNotFound
	}
	delete(f.instances, k)
	if f.onDelete != nil {
		f.onDelete(zone, name)
	}
	return nil
}

// Case 1: The winner's grant triggers deletes of all others.
func TestRacer_WinnerTriggersDeletesOfAllOthers(t *testing.T) {
	backend := newFakeBackend()
	opts := RaceOptions{
		Spec: MachineSpec{
			Name:        "hero-race",
			MachineType: "g4-standard-192",
		},
		Zones:        []string{"us-central1-b", "us-central1-c", "us-east4-b"},
		PollInterval: 10 * time.Millisecond,
	}

	// When us-east4-b is described for the second time, simulate it becoming RUNNING.
	var eastDescribes int
	backend.onDescribe = func(zone, name string, inst *Instance) {
		if zone == "us-east4-b" {
			eastDescribes++
			if eastDescribes >= 2 {
				inst.Status = StatusRunning
			}
		}
	}

	winner, err := Race(t.Context(), backend, opts)
	if err != nil {
		t.Fatalf("Race failed: %v", err)
	}

	if winner == nil {
		t.Fatal("Expected a winner, got nil")
	}
	if winner.Zone != "us-east4-b" {
		t.Errorf("Expected winner in us-east4-b, got %s", winner.Zone)
	}
	if winner.Name != "hero-race-us-east4-b" {
		t.Errorf("Expected winner name hero-race-us-east4-b, got %s", winner.Name)
	}

	// Verify that all other zones were deleted.
	backend.mu.Lock()
	defer backend.mu.Unlock()

	for _, z := range []string{"us-central1-b", "us-central1-c"} {
		k := backend.key(z, DeterministicName("hero-race", z))
		if _, exists := backend.instances[k]; exists {
			t.Errorf("Zone %s was not deleted after winner grant", z)
		}
	}

	// The winner itself must remain in the backend.
	winnerKey := backend.key("us-east4-b", "hero-race-us-east4-b")
	if _, exists := backend.instances[winnerKey]; !exists {
		t.Error("Winner instance was deleted unexpectedly")
	}
}

// Case 2: A late second grant is deleted too.
func TestRacer_LateSecondGrantIsDeletedToo(t *testing.T) {
	backend := newFakeBackend()
	opts := RaceOptions{
		Spec: MachineSpec{
			Name:        "hero-race",
			MachineType: "g4-standard-192",
		},
		Zones:        []string{"zone-a", "zone-b"},
		PollInterval: 10 * time.Millisecond,
	}

	// zone-a becomes RUNNING on the second describe.
	var aDescribes int
	var bDeleteCount int
	backend.onDescribe = func(zone, name string, inst *Instance) {
		if zone == "zone-a" {
			aDescribes++
			if aDescribes >= 2 {
				inst.Status = StatusRunning
			}
		}
	}

	// When zone-b is deleted for the first time, simulate a late grant by marking it RUNNING
	// before the deletion finishes.
	backend.onDelete = func(zone, name string) {
		if zone == "zone-b" {
			bDeleteCount++
			if bDeleteCount == 1 {
				// Re-insert as RUNNING to simulate an in-flight grant during deletion.
				k := backend.key(zone, name)
				backend.instances[k] = &Instance{
					Name:   name,
					Zone:   zone,
					Status: StatusRunning,
				}
			}
		}
	}

	winner, err := Race(t.Context(), backend, opts)
	if err != nil {
		t.Fatalf("Race failed: %v", err)
	}

	if winner.Zone != "zone-a" {
		t.Errorf("Expected winner zone-a, got %s", winner.Zone)
	}

	// Verify zone-b is confirmed deleted despite the late second grant.
	backend.mu.Lock()
	defer backend.mu.Unlock()

	k := backend.key("zone-b", DeterministicName("hero-race", "zone-b"))
	if _, exists := backend.instances[k]; exists {
		t.Error("zone-b was not deleted after its late grant")
	}
	if bDeleteCount < 2 {
		t.Errorf("Expected at least 2 delete attempts for late grant zone-b, got %d", bDeleteCount)
	}
}

// Case 3: Expiry re-requests and unsupported zones are dropped.
func TestRacer_ExpiryReRequestsAndUnsupportedZoneDropped(t *testing.T) {
	backend := newFakeBackend()
	// Mark unsupported-zone as failing with ErrUnsupportedZone.
	backend.failCreate["unsupported-zone"] = ErrUnsupportedZone

	opts := RaceOptions{
		Spec: MachineSpec{
			Name:        "hero-race",
			MachineType: "g4-standard-192",
		},
		Zones:        []string{"unsupported-zone", "expiring-zone"},
		PollInterval: 10 * time.Millisecond,
	}

	var describeCount int
	backend.onDescribe = func(zone, name string, inst *Instance) {
		if zone == "expiring-zone" {
			describeCount++
			if describeCount == 2 {
				// Simulate expiration on the second describe.
				inst.Status = StatusTerminated
				inst.StatusMessage = "flex-start wait time expired"
			} else if describeCount >= 4 {
				// Recreated instance is granted capacity.
				inst.Status = StatusRunning
			}
		}
	}

	winner, err := Race(t.Context(), backend, opts)
	if err != nil {
		t.Fatalf("Race failed: %v", err)
	}

	if winner.Zone != "expiring-zone" {
		t.Errorf("Expected winner in expiring-zone, got %s", winner.Zone)
	}

	backend.mu.Lock()
	defer backend.mu.Unlock()

	// Verify expiring-zone was created at least twice (initial + recreation).
	var createsInExpiring int
	for _, z := range backend.creates {
		if z == "expiring-zone" {
			createsInExpiring++
		}
	}
	if createsInExpiring < 2 {
		t.Errorf("Expected at least 2 creates for expiring-zone, got %d", createsInExpiring)
	}
}

// Case 4: Cancellation on signal (context cancelled).
func TestRacer_CancellationOnSignal(t *testing.T) {
	backend := newFakeBackend()
	ctx, cancel := context.WithCancel(t.Context())

	opts := RaceOptions{
		Spec: MachineSpec{
			Name:        "hero-race",
			MachineType: "g4-standard-192",
		},
		Zones:        []string{"zone-a", "zone-b"},
		PollInterval: 10 * time.Millisecond,
	}

	// Cancel the context after the initial creation has happened.
	backend.onDescribe = func(zone, name string, inst *Instance) {
		cancel()
	}

	winner, err := Race(ctx, backend, opts)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Expected context.Canceled error, got: %v (winner: %+v)", err, winner)
	}

	// Verify all instances were deleted on cancellation.
	backend.mu.Lock()
	defer backend.mu.Unlock()

	if len(backend.instances) != 0 {
		t.Errorf("Expected all instances to be deleted on signal, found %d lingering", len(backend.instances))
	}
}

// Case 5: Idempotent adoption of pre-existing requests.
func TestRacer_IdempotentAdoption(t *testing.T) {
	backend := newFakeBackend()
	baseName := "hero-race"

	// Pre-create instances in zone-a and zone-b.
	nameA := DeterministicName(baseName, "zone-a")
	nameB := DeterministicName(baseName, "zone-b")
	backend.instances[backend.key("zone-a", nameA)] = &Instance{
		Name:   nameA,
		Zone:   zoneA(),
		Status: StatusPending,
	}
	backend.instances[backend.key("zone-b", nameB)] = &Instance{
		Name:   nameB,
		Zone:   "zone-b",
		Status: StatusPending,
	}

	opts := RaceOptions{
		Spec: MachineSpec{
			Name:        baseName,
			MachineType: "g4-standard-192",
		},
		Zones:        []string{"zone-a", "zone-b"},
		PollInterval: 10 * time.Millisecond,
	}

	// zone-a becomes RUNNING on the second describe.
	var aDescribes int
	backend.onDescribe = func(zone, name string, inst *Instance) {
		if zone == "zone-a" {
			aDescribes++
			if aDescribes >= 2 {
				inst.Status = StatusRunning
			}
		}
	}

	winner, err := Race(t.Context(), backend, opts)
	if err != nil {
		t.Fatalf("Race failed: %v", err)
	}

	if winner.Zone != "zone-a" {
		t.Errorf("Expected winner zone-a, got %s", winner.Zone)
	}

	backend.mu.Lock()
	defer backend.mu.Unlock()

	// Verify no new instances were created since existing ones were adopted.
	if len(backend.creates) != 0 {
		t.Errorf("Expected 0 creates due to adoption, got %d (%v)", len(backend.creates), backend.creates)
	}

	// zone-b must be deleted upon zone-a winning.
	if _, exists := backend.instances[backend.key("zone-b", nameB)]; exists {
		t.Error("Pre-existing request in zone-b was not deleted when zone-a won")
	}
}

func zoneA() string {
	return "zone-a"
}

// Verify that a timeout terminates the race and cleans up all requests.
func TestRacer_TimeoutCleansUp(t *testing.T) {
	backend := newFakeBackend()
	opts := RaceOptions{
		Spec: MachineSpec{
			Name:        "hero-race",
			MachineType: "g4-standard-192",
		},
		Zones:        []string{"zone-a", "zone-b"},
		Timeout:      30 * time.Millisecond,
		PollInterval: 5 * time.Millisecond,
	}

	winner, err := Race(t.Context(), backend, opts)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Expected DeadlineExceeded, got: %v (winner: %+v)", err, winner)
	}

	backend.mu.Lock()
	defer backend.mu.Unlock()

	if len(backend.instances) != 0 {
		t.Errorf("Expected all instances deleted on timeout, found %d", len(backend.instances))
	}
}

func TestErrorClassification(t *testing.T) {
	if !IsNotFound(ErrNotFound) {
		t.Error("ErrNotFound should be recognized as not found")
	}
	if !IsNotFound(errors.New("googleapi: Error 404: The resource '...' was not found: notFound")) {
		t.Error("404 error should be recognized as not found")
	}
	if !IsUnsupportedZone(ErrUnsupportedZone) {
		t.Error("ErrUnsupportedZone should be recognized as unsupported zone")
	}
	if !IsUnsupportedZone(errors.New("Machine type g4-standard-192 is not available in zone us-central1-a")) {
		t.Error("unsupported machine type error should be recognized")
	}
}
