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
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
)

// cmdTrace routes `vxpu trace ...` subcommands.
func cmdTrace(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu trace summarize [flags] FILE")
	}
	switch args[0] {
	case "summarize":
		return runTraceSummarize(ctx, args[1:])
	default:
		return fmt.Errorf("vxpu trace: unknown subcommand %q (summarize)", args[0])
	}
}

func runTraceSummarize(ctx context.Context, args []string) (err error) {
	flags := flag.NewFlagSet("trace summarize", flag.ContinueOnError)
	top := flags.Int("top", 5, "number of slowest steps to show")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu trace summarize [flags] FILE")
	}
	path := flags.Arg(0)
	var r io.Reader
	if path == "-" {
		r = os.Stdin
	} else {
		f, openErr := os.Open(path)
		if openErr != nil {
			return openErr
		}
		defer func() {
			if cerr := f.Close(); err == nil {
				err = cerr
			}
		}()
		r = f
	}
	return SummarizeTrace(r, os.Stdout, *top)
}

// StepTrace represents one JSON line written by the native loop (DecodeLoop::trace_step).
type StepTrace struct {
	Step            int64      `json:"step"`
	Decoding        int64      `json:"decoding"`
	Ms              float64    `json:"ms"`
	Launch0Ms       float64    `json:"launch0_ms"`
	LaunchedMs      float64    `json:"launched_ms"`
	SyncedMs        []float64  `json:"synced_ms"`
	DraftLaunch0Ms  *float64   `json:"draft_launch0_ms,omitempty"`
	DraftLaunchedMs *float64   `json:"draft_launched_ms,omitempty"`
	DraftCheckedMs  *float64   `json:"draft_checked_ms,omitempty"`
	DraftHost       []HostNode `json:"draft_host,omitempty"`
	Host            []HostNode `json:"host,omitempty"`
}

// HostNode represents a host node execution start and end times in ms since step start.
type HostNode struct {
	Name    string  `json:"name,omitempty"`
	StartMs float64 `json:"start_ms"`
	EndMs   float64 `json:"end_ms"`
}

// Duration returns the host node duration in milliseconds.
func (h HostNode) Duration() float64 {
	d := h.EndMs - h.StartMs
	if d < 0 {
		return 0
	}
	return d
}

// PhaseValue holds the duration of a phase in one step.
type PhaseValue struct {
	Name     string
	Duration float64
}

func stepPhases(s *StepTrace) []PhaseValue {
	var phases []PhaseValue
	phases = append(phases, PhaseValue{Name: "wall", Duration: s.Ms})

	if s.DraftLaunch0Ms != nil && s.DraftLaunchedMs != nil {
		dLaunch := *s.DraftLaunchedMs - *s.DraftLaunch0Ms
		if dLaunch < 0 {
			dLaunch = 0
		}
		phases = append(phases, PhaseValue{Name: "draft launch", Duration: dLaunch})
	}

	for i, h := range s.DraftHost {
		name := fmt.Sprintf("draft host node %d", i)
		if h.Name != "" {
			if strings.HasPrefix(h.Name, "draft host") {
				name = h.Name
			} else {
				name = fmt.Sprintf("draft host node %d (%s)", i, h.Name)
			}
		}
		phases = append(phases, PhaseValue{Name: name, Duration: h.Duration()})
	}

	launch := s.LaunchedMs - s.Launch0Ms
	if launch < 0 {
		launch = 0
	}
	phases = append(phases, PhaseValue{Name: "launch", Duration: launch})

	for g, syncMs := range s.SyncedMs {
		var drain float64
		if g == 0 {
			drain = syncMs - s.LaunchedMs
		} else {
			drain = syncMs - s.SyncedMs[g-1]
		}
		if drain < 0 {
			drain = 0
		}
		phases = append(phases, PhaseValue{Name: fmt.Sprintf("gpu %d", g), Duration: drain})
	}

	for i, h := range s.Host {
		name := fmt.Sprintf("host node %d", i)
		if h.Name != "" {
			if strings.HasPrefix(h.Name, "host") {
				name = h.Name
			} else {
				name = fmt.Sprintf("host node %d (%s)", i, h.Name)
			}
		}
		phases = append(phases, PhaseValue{Name: name, Duration: h.Duration()})
	}

	return phases
}

// PhaseStats stores summary statistics for a phase across all steps.
type PhaseStats struct {
	Name   string
	P10    float64
	Median float64
	P90    float64
	Max    float64
}

// SummarizeTrace reads a per-step host trace from r and prints a summary to out.
func SummarizeTrace(r io.Reader, out io.Writer, top int) error {
	reader := bufio.NewReader(r)
	var steps []StepTrace
	lineNum := 0
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			lineNum++
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				if err != nil {
					break
				}
				continue
			}
			var step StepTrace
			if unmarshalErr := json.Unmarshal(line, &step); unmarshalErr != nil {
				return fmt.Errorf("line %d: invalid trace JSON: %w (note: early traces on the pod may have dropped a bracket before drafter fields)", lineNum, unmarshalErr)
			}
			steps = append(steps, step)
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("reading trace: %w", err)
		}
	}
	if len(steps) == 0 {
		return fmt.Errorf("trace contains no steps")
	}

	var phaseOrder []string
	phaseSamples := make(map[string][]float64)

	for i := range steps {
		phases := stepPhases(&steps[i])
		for _, p := range phases {
			if _, exists := phaseSamples[p.Name]; !exists {
				phaseOrder = append(phaseOrder, p.Name)
			}
			phaseSamples[p.Name] = append(phaseSamples[p.Name], p.Duration)
		}
	}

	var stats []PhaseStats
	for _, name := range phaseOrder {
		samples := append([]float64(nil), phaseSamples[name]...)
		sort.Float64s(samples)
		stats = append(stats, PhaseStats{
			Name:   name,
			P10:    tracePercentile(samples, 0.10),
			Median: tracePercentile(samples, 0.50),
			P90:    tracePercentile(samples, 0.90),
			Max:    samples[len(samples)-1],
		})
	}

	if _, err := fmt.Fprintf(out, "%d steps:\n\n", len(steps)); err != nil {
		return err
	}
	w := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	if _, err := fmt.Fprintln(w, "PHASE\tP10 (ms)\tMEDIAN (ms)\tP90 (ms)\tMAX (ms)"); err != nil {
		return err
	}
	for _, s := range stats {
		if _, err := fmt.Fprintf(w, "%s\t%.2f\t%.2f\t%.2f\t%.2f\n", s.Name, s.P10, s.Median, s.P90, s.Max); err != nil {
			return err
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}

	if top <= 0 {
		top = 5
	}
	sortedSteps := append([]StepTrace(nil), steps...)
	sort.SliceStable(sortedSteps, func(i, j int) bool {
		if sortedSteps[i].Ms != sortedSteps[j].Ms {
			return sortedSteps[i].Ms > sortedSteps[j].Ms
		}
		return sortedSteps[i].Step < sortedSteps[j].Step
	})

	if top > len(sortedSteps) {
		top = len(sortedSteps)
	}

	if _, err := fmt.Fprintf(out, "\nslowest steps:\n"); err != nil {
		return err
	}
	for i := 0; i < top; i++ {
		st := sortedSteps[i]
		phases := stepPhases(&st)
		var subPhases []PhaseValue
		for _, p := range phases {
			if p.Name == "wall" {
				continue
			}
			subPhases = append(subPhases, p)
		}
		// Sort subphases by duration descending to highlight bottlenecks first.
		sort.SliceStable(subPhases, func(a, b int) bool {
			return subPhases[a].Duration > subPhases[b].Duration
		})
		var parts []string
		for _, p := range subPhases {
			parts = append(parts, fmt.Sprintf("%s: %.2f ms", p.Name, p.Duration))
		}
		if len(parts) > 0 {
			if _, err := fmt.Fprintf(out, "  step %d: wall %.2f ms (%s)\n", st.Step, st.Ms, strings.Join(parts, ", ")); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprintf(out, "  step %d: wall %.2f ms\n", st.Step, st.Ms); err != nil {
				return err
			}
		}
	}

	return nil
}

func tracePercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Min(float64(len(sorted)-1), math.Floor(p*float64(len(sorted)))))
	return sorted[idx]
}
