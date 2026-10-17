# vXPU Results Site Generator

A static HTML results site generator over benchmark and test vector JSON files produced by vXPU development tools.

No server, no background daemon, and no external CSS or JavaScript dependencies. The generated site is viewable offline directly from a `file://` URL or served from any static web host.

## Quick Start

Generate a results site from a directory of benchmark results:

```sh
python vxpu/dev/site/build.py RESULTS_DIR OUT_DIR
```

To build from the bundled fixture results:

```sh
python vxpu/dev/site/build.py vxpu/dev/site/fixtures/results /tmp/vxpu-site
open /tmp/vxpu-site/index.html   # or open in your browser
```

Run tests:

```sh
python vxpu/dev/site/test_build.py
```

## Inputs

`build.py` recursively searches `RESULTS_DIR` for run directories containing:

1. **`attribute_decode.py` outputs** (`vxpu/dev/vm/attribute_decode.py`):
   - Clean decode latency in ms/token (median, best, and individual session timings)
   - Per-step decode time breakdown by segment kind (`dense`, `experts`, `engram`)
   - Expert routing body duration vs dispatch and device crossings
   - Heaviest dense segments with node counts, top ops, and source modules
   - Number of decode steps executed

2. **`python -m vxpu.vectors` recordings** (format `vxpu-vectors/1`) and comparisons:
   - Output vectors recorded by `transformers`, `vxpu-engine`, `vxpu-router`, or `vllm`
   - Greedy generated tokens and top-k logit distributions per prompt
   - Automated comparison against a reference recording (excusing flips within the logit tie set)
   - Optional pre-computed compare JSON files (`compare.json`)

3. **Optional sidecar `run.json`** per run:
   ```json
   {
     "label": "Triton fp8 GEMV",
     "commit": "4a3ce9f",
     "artifact": "deepseek-v41-flash-fp8",
     "hardware": "NVIDIA RTX PRO 6000",
     "date": "2026-10-17T17:00:00Z",
     "notes": "fp8 weight read directly on GPU via Triton GEMV",
     "reference": "../reference_vectors.json"
   }
   ```
   All fields are optional. If omitted:
   - `label` defaults to the run directory name.
   - `date` defaults to directory modification timestamp.
   - `hardware` and `commit` can be inferred from vector recordings if present.
   - `reference` defaults to any `ref.json` or `reference*.json` found in the run directory or at the root of `RESULTS_DIR`.

## Output

A directory containing static HTML files:

- **Index page (`index.html`):**
  - **Latency trajectory chart:** An inline SVG line and area chart plotting median and best ms/token across runs. Drawn with zero JS libraries, with responsive viewBox scaling and hover tooltips.
  - **Runs overview table:** Summary table displaying label, date, hardware, commit hash, median/best ms/token, vector comparison status, and notes, with links to detailed run pages.
  - Overall summary cards showing total runs and best decode rate recorded.

- **Run detail page (`run-<id>.html`):**
  - **Per-step time breakdown bar:** An inline SVG horizontal stacked bar visualizing time spent in dense segments, expert host ops, and engram lookups, along with percentage allocations.
  - **Expert routing body breakout:** Highlights time spent in CPU kernels & glue against dispatch/device crossings inside the expert host op.
  - **Heaviest segments table:** Detailed breakdown of the most expensive dense segments, including duration, percentage of dense step, node count, frequent ops, and originating module hierarchy.
  - **Test vectors verification table:** Evaluates greedy token equivalence against the chosen reference file, reporting tokens compared, first flip position (with tie/outside classification and logit gap), and maximum logit difference (absolute and relative).

## CLI Reference

```sh
python vxpu/dev/site/build.py [-h] [--reference REF_VECTORS_JSON] [--title TITLE] RESULTS_DIR OUT_DIR
```

- `RESULTS_DIR`: Directory containing run results, searched recursively.
- `OUT_DIR`: Target directory where HTML pages are written.
- `--reference REF_VECTORS_JSON` *(optional)*: Global reference `vxpu-vectors/1` file to compare runs against if not specified in `run.json` or run directory.
- `--title TITLE` *(optional)*: Custom title for the generated site (default: "vXPU Performance & Verification Results").

## Design & Aesthetics

- **Zero dependencies:** Embedded Vanilla CSS; no external web fonts, CDNs, or JavaScript.
- **Dark & Light Mode:** Automatic switching based on system preference (`prefers-color-scheme`).
- **File URL Compatible:** Works cleanly over `file://` or static hosting.
