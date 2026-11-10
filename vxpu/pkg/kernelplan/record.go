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

package kernelplan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	RecordFormat        = "vxpu-run-record/1"
	RecordSchemaVersion = 1
)

// RunRecord is the record written by the executor at the end of a run.
type RunRecord struct {
	Format            string             `json:"format"`
	SchemaVersion     int                `json:"schema_version"`
	PlanHash          string             `json:"plan_hash"`
	TreeCommit        string             `json:"tree_commit"`
	ArtifactMD5       string             `json:"artifact_md5"`
	KnobsInEffect     map[string]string  `json:"knobs_in_effect"`
	MedianStepMs      float64            `json:"median_step_ms"`
	TokensPerStep     float64            `json:"tokens_per_step"`
	Checks            map[string]any     `json:"checks,omitempty"`
	PerCardPeakMemory map[string]int64   `json:"per_card_peak_memory,omitempty"`
	BuildTimeS        float64            `json:"build_time_s"`
	PerClassTimes     map[string]float64 `json:"per_class_times,omitempty"`
	Timestamp         string             `json:"timestamp,omitempty"`
}

// ValidateRecord validates a RunRecord ensuring format, schema version and required fields are present.
func ValidateRecord(data []byte) (*RunRecord, error) {
	var rec RunRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("unmarshal run record: %w", err)
	}

	if rec.Format != RecordFormat {
		return nil, fmt.Errorf("unexpected record format %q, expected %q", rec.Format, RecordFormat)
	}
	if rec.SchemaVersion != RecordSchemaVersion {
		return nil, fmt.Errorf("unexpected record schema version %d, expected %d", rec.SchemaVersion, RecordSchemaVersion)
	}
	if rec.PlanHash == "" {
		return nil, errors.New("missing plan_hash")
	}
	if rec.TreeCommit == "" {
		return nil, errors.New("missing tree_commit")
	}
	if rec.ArtifactMD5 == "" {
		return nil, errors.New("missing artifact_md5")
	}
	if rec.KnobsInEffect == nil {
		return nil, errors.New("missing knobs_in_effect")
	}
	return &rec, nil
}

// AppendLedger appends a validated RunRecord to the specified ledger file as a single JSON line.
func AppendLedger(ledgerPath string, rec *RunRecord) (err error) {
	if rec.Timestamp == "" {
		rec.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshal run record: %w", err)
	}

	if dir := filepath.Dir(ledgerPath); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}

	f, err := os.OpenFile(ledgerPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open ledger %s: %w", ledgerPath, err)
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()

	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("write ledger line: %w", err)
	}
	return nil
}
