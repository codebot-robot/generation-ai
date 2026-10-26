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

package weightdisk

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fakeGcloud records the commands and answers the factory's guest
// attributes: copying for the first poll, then done.
type fakeGcloud struct {
	calls         []string
	polls         int
	fail          map[string]bool // command prefixes that fail
	metadataFiles map[string][]byte
}

func (f *fakeGcloud) run(_ context.Context, args ...string) ([]byte, error) {
	cmd := strings.Join(args, " ")
	f.calls = append(f.calls, cmd)
	for prefix := range f.fail {
		if strings.HasPrefix(cmd, prefix) {
			return nil, fmt.Errorf("%s failed", prefix)
		}
	}
	for i, arg := range args {
		if arg == "--metadata-from-file" && i+1 < len(args) {
			if f.metadataFiles == nil {
				f.metadataFiles = make(map[string][]byte)
			}
			for _, pair := range strings.Split(args[i+1], ",") {
				k, p, ok := strings.Cut(pair, "=")
				if ok {
					if data, err := os.ReadFile(p); err == nil {
						f.metadataFiles[k] = data
					}
				}
			}
		}
	}
	if strings.HasPrefix(cmd, "compute instances get-guest-attributes") {
		f.polls++
		if f.polls == 1 {
			return []byte(`[{"namespace":"vxpu","key":"status","value":"copying"}]`), nil
		}
		return []byte(`[{"namespace":"vxpu","key":"status","value":"done"},{"namespace":"vxpu","key":"copy_seconds","value":"412"}]`), nil
	}
	return nil, nil
}

func (f *fakeGcloud) find(prefix string) string {
	for _, c := range f.calls {
		if strings.HasPrefix(c, prefix) {
			return c
		}
	}
	return ""
}

func TestBuild(t *testing.T) {
	g := &fakeGcloud{}
	res, err := Build(t.Context(), g.run, BuildOptions{
		Project: "p", Zone: "us-central1-a", Image: "ds-flash", Bucket: "gs://weights",
		Files: map[string]int64{"aa": 300e9, "bb": 200e9}, Poll: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Files != 2 || res.Bytes != 500e9 || res.CopySeconds != 412 {
		t.Errorf("result %+v", res)
	}
	create := g.find("compute instances create ds-flash-factory")
	for _, want := range []string{
		"--create-disk name=ds-flash-build,size=535,type=hyperdisk-balanced,device-name=vxpu-weights,auto-delete=no,provisioned-throughput=1200,provisioned-iops=4800",
		"--machine-type c3d-standard-30", "vxpu-bucket=gs://weights", "enable-guest-attributes=TRUE",
		"vxpu-index=",
	} {
		if !strings.Contains(create, want) {
			t.Errorf("factory create %q lacks %q", create, want)
		}
	}
	indexBytes, ok := g.metadataFiles["vxpu-index"]
	if !ok {
		t.Fatal("factory create lacks vxpu-index metadata")
	}
	var index Index
	if err := json.Unmarshal(indexBytes, &index); err != nil {
		t.Fatalf("unmarshal vxpu-index: %v", err)
	}
	if index.Format != Format {
		t.Errorf("index format %q, want %q", index.Format, Format)
	}
	if index.Source != "gs://weights" {
		t.Errorf("index source %q, want gs://weights", index.Source)
	}
	wantFiles := map[string]int64{"aa": 300e9, "bb": 200e9}
	if !reflect.DeepEqual(index.Files, wantFiles) {
		t.Errorf("index files %+v, want %+v", index.Files, wantFiles)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, IndexFile), indexBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	read, err := ReadIndex(dir)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	if !reflect.DeepEqual(read, &index) {
		t.Errorf("ReadIndex = %+v, want %+v", read, index)
	}
	if strings.Contains(factoryScript, "python") {
		t.Errorf("factoryScript still contains python: %s", factoryScript)
	}
	// the VM goes before the image is made from its disk; the disk after
	order := []string{"compute instances create", "compute instances get-guest-attributes", "compute instances delete ds-flash-factory",
		"compute snapshots create ds-flash", "compute disks delete ds-flash-build"}
	at := 0
	for _, c := range g.calls {
		if at < len(order) && strings.HasPrefix(c, order[at]) {
			at++
		}
	}
	if at != len(order) {
		t.Errorf("commands out of order:\n%s", strings.Join(g.calls, "\n"))
	}
	if g.polls != 2 {
		t.Errorf("polled %d times, want 2", g.polls)
	}
}

func TestBuildDeletesTheFactoryWhenTheCopyFails(t *testing.T) {
	g := &fakeGcloud{}
	failed := func(ctx context.Context, args ...string) ([]byte, error) {
		if strings.HasPrefix(strings.Join(args, " "), "compute instances get-guest-attributes") {
			return []byte(`[{"key":"status","value":"failed: gcloud storage cp: 403"}]`), nil
		}
		return g.run(ctx, args...)
	}
	_, err := Build(t.Context(), failed, BuildOptions{Project: "p", Zone: "z", Image: "x", Bucket: "gs://b",
		Files: map[string]int64{"aa": 1}, Poll: time.Millisecond})
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Errorf("err %v, want the factory's failure", err)
	}
	if g.find("compute instances delete x-factory") == "" || g.find("compute snapshots create") != "" {
		t.Errorf("want the factory deleted and no image:\n%s", strings.Join(g.calls, "\n"))
	}
	if g.find("compute disks delete x-build") == "" {
		t.Errorf("want the factory's disk deleted too (it bills by the hour):\n%s", strings.Join(g.calls, "\n"))
	}
}

func TestReadIndex(t *testing.T) {
	dir := t.TempDir()
	idx := Index{
		Format: Format,
		Source: "gs://test-bucket",
		Files:  map[string]int64{"blob1": 1024, "blob2": 2048},
	}
	data, err := json.Marshal(idx)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, IndexFile), data, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadIndex(dir)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	if !reflect.DeepEqual(got, &idx) {
		t.Errorf("ReadIndex = %+v, want %+v", got, idx)
	}

	// wrong format
	badFormat, _ := json.Marshal(Index{Format: "other-format/1", Source: "gs://b", Files: idx.Files})
	if err := os.WriteFile(filepath.Join(dir, IndexFile), badFormat, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadIndex(dir); err == nil || !strings.Contains(err.Error(), "format") {
		t.Errorf("ReadIndex wrong format: err %v, want format error", err)
	}

	// invalid JSON
	if err := os.WriteFile(filepath.Join(dir, IndexFile), []byte("not-json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadIndex(dir); err == nil {
		t.Error("ReadIndex invalid JSON: want error")
	}

	// not a weights disk (missing weights.json)
	if _, err := ReadIndex(t.TempDir()); err == nil {
		t.Error("ReadIndex missing file: want error")
	}
}
