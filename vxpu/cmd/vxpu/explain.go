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

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
)

// explainOffline ranks the ways a model can run on a set of machines
// (planner.Candidates, #409) without running anything: the artifact says
// what the model reads, the machine catalog what each machine does, the
// cost model what each layout costs.
func explainOffline(ctx context.Context, artifactDir, fleetSpec, recordsPath, cutsPath string, asJSON bool, serve string) error {
	if artifactDir == "" {
		return fmt.Errorf("--machines needs --artifact")
	}
	manifest, err := os.ReadFile(filepath.Join(artifactDir, "manifest.json"))
	if err != nil {
		return err
	}
	binding, err := os.ReadFile(filepath.Join(artifactDir, "binding.json"))
	if err != nil {
		return err
	}
	a, err := planner.ReadArtifact(manifest, binding)
	if err != nil {
		return err
	}
	if cutsPath == "" { // the artifact's own, if it has one
		switch _, err := os.Stat(filepath.Join(artifactDir, "cuts.json")); {
		case err == nil:
			cutsPath = filepath.Join(artifactDir, "cuts.json")
		case !os.IsNotExist(err):
			return err
		}
	}
	if cutsPath != "" { // where the step can be cut into ranges (python -m vxpu.cuts)
		data, err := os.ReadFile(cutsPath)
		if err != nil {
			return err
		}
		if err := planner.ReadCuts(&a, data); err != nil {
			return err
		}
	}
	fleet, err := parseFleet(fleetSpec)
	if err != nil {
		return err
	}
	var counts map[int]map[int]int64
	if recordsPath != "" {
		if counts, err = decodeSelections(recordsPath); err != nil {
			return err
		}
	}
	cands := planner.Candidates(a, fleet, counts)
	if serve != "" {
		var src struct {
			Source struct {
				Repo string `json:"repo"`
			} `json:"source"`
		}
		if err := json.Unmarshal(manifest, &src); err != nil {
			return fmt.Errorf("manifest: %w", err)
		}
		return serveSearch(ctx, serve, src.Source.Repo, a, fleet, counts, cands)
	}
	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(cands)
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	if _, err := fmt.Fprintln(w, "RANK\tMS/TOKEN\tKIND\tLAYOUT\tRUNNABLE"); err != nil {
		return err
	}
	for i, c := range cands {
		ms, run := fmt.Sprintf("%.1f", c.Prediction.MsPerToken), "yes"
		if c.Infeasible != "" {
			ms, run = "-", "no: "+c.Infeasible
		} else if !c.Runnable {
			run = "not yet: " + c.NotYet
		}
		if _, err := fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%s\n", i+1, ms, c.Kind, c.Summary, run); err != nil {
			return err
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}
	for i, c := range cands {
		if c.Infeasible != "" {
			continue
		}
		fmt.Printf("\n%d. %s: %.1f ms/token\n", i+1, c.Summary, c.Prediction.MsPerToken)
		fmt.Printf("   parts (ms):%s\n", formatParts(c.Prediction.Parts))
		for _, b := range c.Prediction.Basis {
			fmt.Printf("   %s\n", b)
		}
	}
	return nil
}

func formatParts(parts map[string]float64) string {
	keys := make([]string, 0, len(parts))
	for k := range parts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, " %s %.1f", k, parts[k])
	}
	return b.String()
}

// parseFleet reads TYPE[:COUNT],... into catalog machines, named TYPE#N.
func parseFleet(spec string) ([]planner.Node, error) {
	var fleet []planner.Node
	for _, item := range strings.Split(spec, ",") {
		name, count := strings.TrimSpace(item), 1
		if i := strings.LastIndex(name, ":"); i >= 0 {
			n, err := strconv.Atoi(name[i+1:])
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--machines %q: a count must be a positive number", item)
			}
			name, count = name[:i], n
		}
		m, err := machines.Lookup(name)
		if err != nil {
			return nil, err
		}
		for range count {
			fleet = append(fleet, planner.Node{Name: fmt.Sprintf("%s#%d", m.Name, len(fleet)+1), Probe: machines.ProbeOf(m)})
		}
	}
	return fleet, nil
}

// decodeSelections sums the decode expert selections of request records
// (one JSON object a line, as vxpu stats --records writes them).
func decodeSelections(path string) (_ map[int]map[int]int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	counts := map[int]map[int]int64{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<20), 64<<20)
	for sc.Scan() {
		if len(bytes.TrimSpace(sc.Bytes())) == 0 {
			continue
		}
		var rec struct {
			Telemetry struct {
				Experts struct {
					Decode map[string]map[string]int64 `json:"decode"`
				} `json:"experts"`
			} `json:"telemetry"`
		}
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		for l, experts := range rec.Telemetry.Experts.Decode {
			layer, err := strconv.Atoi(l)
			if err != nil {
				return nil, fmt.Errorf("%s: layer %q: %w", path, l, err)
			}
			if counts[layer] == nil {
				counts[layer] = map[int]int64{}
			}
			for e, n := range experts {
				expert, err := strconv.Atoi(e)
				if err != nil {
					return nil, fmt.Errorf("%s: layer %d expert %q: %w", path, layer, e, err)
				}
				counts[layer][expert] += n
			}
		}
	}
	return counts, sc.Err()
}
