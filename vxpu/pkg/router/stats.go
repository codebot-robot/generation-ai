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

package router

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// Statistics about models served by name (#357): the engine's record of
// how it behaves in the real world, for us and, later, for the planner
// (#316, #349).
//
// Every Chat and Generate on a session opened by model name is recorded:
// the request's statistics (tokens, prefill, ms/token), which commit and
// artifact served it and on which executor, and, for a sampled share of
// requests (every Nth; --telemetry-every), the executor's execution
// telemetry (segment times, host-op phases, expert selections, the plan;
// #337). The router turns telemetry on for sampled requests and keeps the
// record to itself unless the client asked for it. Records are batched
// into JSON-lines blobs in modelstore, one tag per UTC day on the
// repository vxpu-stats/<model>, so they outlive the router; GetModelStats
// summarises them.

const (
	requestFormat     = "vxpu-request/1"
	modelStatsFormat  = "vxpu-model-stats/1"
	statsArtifactType = "application/vnd.gke-labs.vxpu.stats.v1+json"
	statsBatch        = 50
	statsFlushEvery   = 30 * time.Second
)

// modelRef names what serves a model key loaded by name.
type modelRef struct {
	Model, Commit, Artifact string
}

// requestRecord is one request's entry in the statistics.
type requestRecord struct {
	Format          string          `json:"format"`
	Time            string          `json:"time"`
	Model           string          `json:"model"`
	Commit          string          `json:"commit"`
	Artifact        string          `json:"artifact"`
	Executor        string          `json:"executor"`
	RPC             string          `json:"rpc"`
	PromptTokens    int32           `json:"prompt_tokens"`
	PrefilledTokens int32           `json:"prefilled_tokens"`
	Generated       int32           `json:"generated"`
	PrefillMs       float32         `json:"prefill_ms"`
	MsPerToken      float32         `json:"ms_per_token"`
	FinishReason    string          `json:"finish_reason,omitempty"`
	Telemetry       json.RawMessage `json:"telemetry,omitempty"`
}

func statsRepository(model string) string { return "vxpu-stats/" + strings.ToLower(model) }

// statsStore batches records into modelstore.
type statsStore struct {
	ws             *weightstore.Client
	telemetryEvery int64
	requests       atomic.Int64
	records        chan requestRecord
	flushes        chan flushRequest
}

type flushRequest struct {
	model string
	done  chan error
}

func newStatsStore(ws *weightstore.Client, telemetryEvery int) *statsStore {
	st := &statsStore{ws: ws, telemetryEvery: int64(telemetryEvery),
		records: make(chan requestRecord, 1000), flushes: make(chan flushRequest)}
	go st.loop(context.Background())
	return st
}

// sampleTelemetry says whether this request records execution telemetry.
func (st *statsStore) sampleTelemetry() bool {
	if st.telemetryEvery <= 0 {
		return false
	}
	return (st.requests.Add(1)-1)%st.telemetryEvery == 0
}

func (st *statsStore) record(ctx context.Context, r requestRecord) {
	select {
	case st.records <- r:
	default:
		klog.FromContext(ctx).Info("Statistics buffer full; dropping a record", "model", r.Model)
	}
}

// flush writes model's buffered records now (GetModelStats does, so a
// summary includes the latest requests).
func (st *statsStore) flush(ctx context.Context, model string) error {
	req := flushRequest{model: model, done: make(chan error, 1)}
	select {
	case st.flushes <- req:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case err := <-req.done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (st *statsStore) loop(ctx context.Context) {
	log := klog.FromContext(ctx)
	buffered := map[string][]requestRecord{}
	write := func(model string) error {
		recs := buffered[model]
		if len(recs) == 0 {
			return nil
		}
		if err := st.write(ctx, model, recs); err != nil {
			if len(recs) > 10*statsBatch { // modelstore unreachable for long: keep the newest
				buffered[model] = recs[len(recs)-10*statsBatch:]
			}
			return err
		}
		delete(buffered, model)
		return nil
	}
	tick := time.NewTicker(statsFlushEvery)
	defer tick.Stop()
	for {
		select {
		case r := <-st.records:
			buffered[r.Model] = append(buffered[r.Model], r)
			if len(buffered[r.Model]) >= statsBatch {
				if err := write(r.Model); err != nil {
					log.Error(err, "writing statistics", "model", r.Model)
				}
			}
		case req := <-st.flushes:
			for drained := false; !drained; { // records sent before the flush was asked for
				select {
				case r := <-st.records:
					buffered[r.Model] = append(buffered[r.Model], r)
				default:
					drained = true
				}
			}
			req.done <- write(req.model)
		case <-tick.C:
			for model := range buffered {
				if err := write(model); err != nil {
					log.Error(err, "writing statistics", "model", model)
				}
			}
		}
	}
}

// write appends one batch to today's manifest of the model's statistics.
func (st *statsStore) write(ctx context.Context, model string, recs []requestRecord) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, r := range recs {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	name := statsRepository(model)
	digest, err := st.ws.PutBlob(ctx, name, buf.Bytes())
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	tag := "d" + now.Format("20060102")
	m, err := st.ws.GetManifest(ctx, name, tag)
	if err == weightstore.ErrNotFound {
		m = &weightstore.Manifest{SchemaVersion: 2, MediaType: "application/vnd.oci.image.manifest.v1+json",
			ArtifactType: statsArtifactType, Config: weightstore.EmptyConfig,
			Annotations: map[string]string{"vnd.gke-labs.vxpu.model": model}}
	} else if err != nil {
		return err
	}
	m.Layers = append(m.Layers, weightstore.Layer{MediaType: weightstore.MediaTypeFile, Digest: digest, Size: int64(buf.Len()),
		Annotations: map[string]string{"org.opencontainers.image.title": "batch-" + strconv.FormatInt(now.UnixNano(), 10) + ".jsonl"}})
	return st.ws.PutManifest(ctx, name, tag, m)
}

// UseStats records statistics for models served by name, with execution
// telemetry on every Nth request (0: none). Needs modelstore.
func (s *Server) UseStats(telemetryEvery int) {
	if s.weightStore != nil {
		s.stats = newStatsStore(s.weightStore, telemetryEvery)
	}
}

// statsTarget returns what serves modelKey if its requests are recorded.
func (s *Server) statsTarget(modelKey string) (modelRef, string, bool) {
	if s.stats == nil {
		return modelRef{}, "", false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ref, ok := s.keyModel[modelKey]
	if !ok {
		return modelRef{}, "", false
	}
	executor := s.modelEndpoint[modelKey]
	if executor == "" {
		executor = "pod"
	}
	return ref, executor, true
}

func newRecord(ref modelRef, executor, rpc string) requestRecord {
	return requestRecord{Format: requestFormat, Time: time.Now().UTC().Format(time.RFC3339Nano),
		Model: ref.Model, Commit: ref.Commit, Artifact: ref.Artifact, Executor: executor, RPC: rpc}
}

// GetModelStats summarises a model's recorded statistics.
func (s *Server) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
	if s.stats == nil {
		return nil, status.Error(codes.FailedPrecondition, "statistics need modelstore")
	}
	if err := s.stats.flush(ctx, req.Model); err != nil {
		klog.FromContext(ctx).Error(err, "flushing statistics before the summary", "model", req.Model)
	}
	name := statsRepository(req.Model)
	tags, err := s.weightStore.ListTags(ctx, name)
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "%v", err)
	}
	sort.Strings(tags)
	var lines [][]byte
	for _, tag := range tags {
		m, err := s.weightStore.GetManifest(ctx, name, tag)
		if err != nil {
			return nil, status.Errorf(codes.Unavailable, "%v", err)
		}
		for _, layer := range m.Layers {
			data, err := s.weightStore.FetchBlob(ctx, name, layer.Digest)
			if err != nil {
				return nil, status.Errorf(codes.Unavailable, "%v", err)
			}
			sc := bufio.NewScanner(bytes.NewReader(data))
			sc.Buffer(make([]byte, 1<<20), 64<<20)
			for sc.Scan() {
				if line := bytes.TrimSpace(sc.Bytes()); len(line) > 0 {
					lines = append(lines, append([]byte(nil), line...))
				}
			}
		}
	}
	summary, err := summarize(req.Model, lines)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}
	resp := &pb.GetModelStatsResponse{SummaryJson: summary}
	if req.Records {
		resp.RecordsJsonl = string(bytes.Join(lines, []byte("\n")))
	}
	return resp, nil
}

// telemetrySummary is the part of a vxpu-telemetry/1 record the summary reads.
type telemetrySummary struct {
	ByKind  map[string]float64                       `json:"by_kind_ms_per_step"`
	HostOps map[string]map[string]map[string]float64 `json:"host_ops"`
	Experts map[string]map[string]map[string]int64   `json:"experts"`
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	return sorted[int(math.Min(float64(len(sorted)-1), math.Floor(p*float64(len(sorted)))))]
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

// summarize turns records (JSON lines) into a vxpu-model-stats/1 summary.
func summarize(model string, lines [][]byte) (string, error) {
	var msPerToken, prefill, promptTokens []float64
	byRPC, commits, executors := map[string]int{}, map[string]bool{}, map[string]bool{}
	byKind, moe := map[string]float64{}, map[string]float64{}
	experts := map[string]map[int64]int64{} // decode: layer -> expert -> count
	var generated int64
	var sampled int
	first, last := "", ""
	for _, line := range lines {
		var r requestRecord
		if err := json.Unmarshal(line, &r); err != nil {
			return "", fmt.Errorf("record: %w", err)
		}
		if first == "" || r.Time < first {
			first = r.Time
		}
		if r.Time > last {
			last = r.Time
		}
		byRPC[r.RPC]++
		commits[r.Commit] = true
		executors[r.Executor] = true
		generated += int64(r.Generated)
		if r.Generated > 1 && r.MsPerToken > 0 {
			msPerToken = append(msPerToken, float64(r.MsPerToken))
		}
		prefill = append(prefill, float64(r.PrefillMs))
		promptTokens = append(promptTokens, float64(r.PromptTokens))
		if len(r.Telemetry) == 0 {
			continue
		}
		var t telemetrySummary
		if err := json.Unmarshal(r.Telemetry, &t); err != nil {
			continue
		}
		sampled++
		for k, v := range t.ByKind {
			byKind[k] += v
		}
		for k, v := range t.HostOps["decode"]["moe_experts"] {
			moe[k] += v
		}
		for layer, counts := range t.Experts["decode"] {
			if experts[layer] == nil {
				experts[layer] = map[int64]int64{}
			}
			for e, n := range counts {
				id, err := strconv.ParseInt(e, 10, 64)
				if err != nil {
					continue // not an expert index; the record is from another format
				}
				experts[layer][id] += n
			}
		}
	}
	sort.Float64s(msPerToken)
	sort.Float64s(prefill)
	sort.Float64s(promptTokens)
	mean := func(xs []float64) float64 {
		if len(xs) == 0 {
			return 0
		}
		sum := 0.0
		for _, x := range xs {
			sum += x
		}
		return sum / float64(len(xs))
	}
	keys := func(m map[string]bool) []string {
		out := make([]string, 0, len(m))
		for k := range m {
			out = append(out, k)
		}
		sort.Strings(out)
		return out
	}
	out := map[string]any{
		"format": modelStatsFormat, "model": model, "requests": len(lines), "by_rpc": byRPC,
		"commits": keys(commits), "executors": keys(executors), "first": first, "last": last,
		"generated_tokens":  generated,
		"ms_per_token":      map[string]float64{"mean": round3(mean(msPerToken)), "p50": round3(percentile(msPerToken, 0.5)), "p90": round3(percentile(msPerToken, 0.9))},
		"prefill_ms":        map[string]float64{"p50": round3(percentile(prefill, 0.5)), "p90": round3(percentile(prefill, 0.9))},
		"prompt_tokens":     map[string]float64{"p50": percentile(promptTokens, 0.5), "max": percentile(promptTokens, 1)},
		"telemetry_sampled": sampled,
	}
	if sampled > 0 {
		for k := range byKind {
			byKind[k] = round3(byKind[k] / float64(sampled))
		}
		for k := range moe {
			moe[k] = round3(moe[k] / float64(sampled))
		}
		out["by_kind_ms_per_step"] = byKind
		out["moe_experts_per_step"] = moe
		out["experts_decode"] = expertConcentration(experts)
	}
	b, err := json.MarshalIndent(out, "", " ")
	return string(b), err
}

// expertConcentration: how much of the decode traffic the hottest
// fraction of each layer's experts carried (an oracle cache's hit rate;
// vxpu.expert_stats has the full analysis).
func expertConcentration(experts map[string]map[int64]int64) map[string]any {
	var maxID, total int64
	for _, counts := range experts {
		for id, n := range counts {
			if id > maxID {
				maxID = id
			}
			total += n
		}
	}
	perLayer := maxID + 1
	out := map[string]any{"selections": total, "layers": len(experts), "experts_per_layer_seen": perLayer}
	for _, f := range []float64{0.25, 0.5} {
		var hit int64
		for _, counts := range experts {
			ns := make([]int64, 0, len(counts))
			for _, n := range counts {
				ns = append(ns, n)
			}
			sort.Slice(ns, func(i, j int) bool { return ns[i] > ns[j] })
			k := int(math.Ceil(f * float64(perLayer)))
			for i := 0; i < k && i < len(ns); i++ {
				hit += ns[i]
			}
		}
		if total > 0 {
			out[fmt.Sprintf("hottest_%d%%_share", int(f*100))] = round3(float64(hit) / float64(total))
		}
	}
	return out
}
