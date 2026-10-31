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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) *Record {
	t.Helper()
	r, err := ReadRecordFile(filepath.Join("testdata", "record-rtx6000.json"))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestPlanChoosesTheMeasuredFastestBody(t *testing.T) {
	r := fixture(t)
	table, err := Plan(t.Context(), []*Record{r}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if table.GPU != r.Hardware.GPU {
		t.Errorf("table.GPU = %q, want the record's", table.GPU)
	}
	want := map[string]string{ // op + shapes -> body
		"vxpu::attn_q rows=1 dim=5120 rank=1280 heads=64x512": "gpu-triton",
		"vxpu::window_kv rows=1 dim=5120 out=512":             "gpu-triton",
		"vxpu::hc_front rows=1 hc=4 dim=5120":                 "gpu-triton-split",
		"vxpu::fp8_linear 1x6144->25600":                      "gpu-fp8-gemv",
		"vxpu::fp8_linear 1x1280->4096":                       "gpu-fp8-gemv",
		// rotary: the kernel measured 2.1 us, the fused body 3.4 us less the 2.0 us launch it saves = 1.4: fused wins
		"vxpu::rotary 1x1x64 rope=64": "gpu-fused",
		// not measured: as the engine ran them
		"vxpu::sparse_attn q=1x64x512 top=2048": "gpu-triton",
		"vxpu::head_linear 1x5120->129280":      "gpu-triton-bf16",
	}
	got := map[string]string{}
	for _, e := range table.Sites {
		if e.Site != "" {
			t.Errorf("entry per site where the sites agree: %+v", e)
		}
		got[e.Op+" "+e.Shapes] = e.Impl
	}
	for k, body := range want {
		if got[k] != body {
			t.Errorf("%s: planned %q, want %q", k, got[k], body)
		}
	}
	if len(got) != len(want) {
		t.Errorf("planned %d entries, want %d: %v", len(got), len(want), got)
	}
	for _, e := range table.Sites {
		if e.Op == "vxpu::moe_experts" {
			t.Errorf("a kept (host) op must not get an entry: %+v", e)
		}
	}
	// the knobs: native fp4 from the card (12.0 is Blackwell), GEMV_ROWS_MAX from the sweep, the rest as run
	if table.Knobs["FP4_NATIVE"] != true {
		t.Errorf("FP4_NATIVE = %v, want true for capability 12.0", table.Knobs["FP4_NATIVE"])
	}
	if table.Knobs["GEMV_ROWS_MAX"] != 64 {
		t.Errorf("GEMV_ROWS_MAX = %v, want 64 (the sweep's fastest)", table.Knobs["GEMV_ROWS_MAX"])
	}
	if table.Knobs["FP4_DIRECT_MAX"] != float64(8) {
		t.Errorf("FP4_DIRECT_MAX = %v (%T), want 8 as the record ran", table.Knobs["FP4_DIRECT_MAX"], table.Knobs["FP4_DIRECT_MAX"])
	}
	if _, set := table.Knobs["FP4_KSPLIT"]; set {
		t.Errorf("a knob the record left to the kernel must stay unset: FP4_KSPLIT = %v", table.Knobs["FP4_KSPLIT"])
	}
	if len(table.Explain) == 0 || !strings.Contains(strings.Join(table.Explain, "\n"), "fp8_linear 1x6144->25600 (2 sites): measured") {
		t.Errorf("the explanation should say what was measured per group: %v", table.Explain)
	}
	// the table round-trips through the engine's format and matches by the engine's rule
	data, err := table.Write()
	if err != nil {
		t.Fatal(err)
	}
	back, err := ReadTable(data)
	if err != nil {
		t.Fatal(err)
	}
	if e := back.Entry("vxpu::fp8_linear", "model.layers.14.engram.wkv", "1x6144->25600"); e == nil || e.Impl != "gpu-fp8-gemv" {
		t.Errorf("Entry for the engram site = %+v", e)
	}
	if e := back.Entry("vxpu::fp8_linear", "model.layers.9.attn.wo", "9x9->9"); e != nil {
		t.Errorf("an op at shapes the table does not name should match nothing, got %+v", e)
	}
}

func TestPlanNamesSitesWhenTheRecordDisagrees(t *testing.T) {
	r := fixture(t)
	// a run with a table given before ran the fused body at one engram site: the planner still
	// takes the measurement, but names every site so the disagreement is visible
	for i := range r.Sites {
		if r.Sites[i].Site == "model.layers.14.engram.wkv" {
			r.Sites[i].Chosen = "gpu-fused"
			r.Sites[i].Source = "table"
		}
	}
	table, err := Plan(t.Context(), []*Record{r}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, e := range table.Sites {
		if e.Op == "vxpu::fp8_linear" && e.Site != "" {
			n++
			if e.Impl != "gpu-fp8-gemv" {
				t.Errorf("%s: %s, want the measured body", e.Site, e.Impl)
			}
		}
	}
	if n != 2 {
		t.Errorf("the two engram sites should each get an entry, got %d", n)
	}
}

func TestPlanRefusesMixedGPUs(t *testing.T) {
	a, b := fixture(t), fixture(t)
	b.Hardware.GPU = "NVIDIA L4"
	if _, err := Plan(t.Context(), []*Record{a, b}, Options{}); err == nil {
		t.Error("records of two GPUs should be refused")
	}
}

func TestCheckFindsABodyTheTableDidNotName(t *testing.T) {
	r := fixture(t)
	table, err := Plan(t.Context(), []*Record{r}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if bad := Check(table, r); len(bad) != 0 {
		t.Errorf("the record the table was planned from should check clean: %v", bad)
	}
	// #490's shape: the lowering chose the GEMV, the step ran the fused body at the engram sites
	ran := fixture(t)
	for i := range ran.Sites {
		if strings.HasSuffix(ran.Sites[i].Site, "engram.wkv") {
			ran.Sites[i].Chosen = "gpu-fused"
		}
	}
	bad := Check(table, ran)
	if len(bad) != 2 || bad[0].Want != "gpu-fp8-gemv" || !strings.Contains(bad[0].String(), "engram.wkv ran gpu-fused, the table names gpu-fp8-gemv") {
		t.Errorf("Check = %v", bad)
	}
}

func TestReadTableRefusesWhatTheEngineRefuses(t *testing.T) {
	for _, bad := range []string{
		`{"format": "nope", "sites": []}`,
		`{"format": "vxpu-kernel-table/1", "sites": [{"op": "vxpu::rotary"}]}`,
		`{"format": "vxpu-kernel-table/1", "knobs": {"NO_SUCH": 1}, "sites": []}`,
		`{"format": "vxpu-kernel-table/1", "knobs": {"GEMV_ROWS_MAX": "16"}, "sites": []}`,
		`{"format": "vxpu-kernel-table/1", "knobs": {"FP4_NATIVE": 1}, "sites": []}`,
		`{"format": "vxpu-kernel-table/1", "sites": [{"op": "vxpu::rotary", "impl": "a"}, {"op": "vxpu::rotary", "impl": "b"}]}`,
	} {
		if _, err := ReadTable([]byte(bad)); err == nil {
			t.Errorf("should be refused: %s", bad)
		}
	}
	good := `{"format": "vxpu-kernel-table/1", "knobs": {"GEMV_ROWS_MAX": 16, "FP4_NATIVE": false},
	          "sites": [{"op": "vxpu::rotary", "impl": "reference"}, {"op": "vxpu::rotary", "site": "model.layers.1.attn", "impl": "gpu-triton"}]}`
	table, err := ReadTable([]byte(good))
	if err != nil {
		t.Fatal(err)
	}
	if e := table.Entry("vxpu::rotary", "model.layers.1.attn", "x"); e.Impl != "gpu-triton" {
		t.Errorf("the site's entry should win: %+v", e)
	}
	if e := table.Entry("vxpu::rotary", "model.layers.2.attn", "x"); e.Impl != "reference" {
		t.Errorf("the op's entry should apply elsewhere: %+v", e)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "t.json")
	data, _ := table.Write()
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadTableFile(path); err != nil {
		t.Fatal(err)
	}
}
