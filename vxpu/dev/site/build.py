# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static results site generator from attribute_decode and vectors JSON files.

Usage:
    python vxpu/dev/site/build.py RESULTS_DIR OUT_DIR [--reference REF_VECTORS_JSON]

Scans RESULTS_DIR recursively for:
  - attribute_decode.py outputs (clean ms/token, per_step_ms, dense_segments)
  - vxpu.vectors recordings (format: vxpu-vectors/1) and compare results
  - optional sidecar run.json: {label, commit, artifact, hardware, date, notes}

Generates a standalone static HTML site (one index page + one page per run)
with zero external CSS/JS dependencies, viewable offline or from file:// URLs.
"""

import argparse
from dataclasses import dataclass, field
import html
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


# --- Data Models ---


@dataclass
class Run:
    id: str
    dir_path: str
    label: str
    commit: Optional[str] = None
    artifact: Optional[str] = None
    hardware: Optional[str] = None
    date: Optional[str] = None
    notes: Optional[str] = None

    # attribute_decode metrics
    clean_ms_per_token: Optional[float] = None
    clean_best_ms_per_token: Optional[float] = None
    clean_sessions_ms_per_token: Optional[List[float]] = None
    decode_steps: Optional[int] = None
    instrumented_ms_per_token: Optional[float] = None
    per_step_ms: Dict[str, float] = field(default_factory=dict)
    experts_routing_body_ms: Optional[float] = None
    dense_segments: List[Dict[str, Any]] = field(default_factory=list)

    # test vectors / comparison
    vectors_reference_name: Optional[str] = None
    vectors_rows: List[Dict[str, Any]] = field(default_factory=list)
    vectors_ok: bool = True

    # raw candidate vector data if uncompared
    raw_vectors: Optional[Dict[str, Any]] = None


# --- Vectors Comparison Logic ---


def _scale(values: List[float]) -> float:
    """The scale a relative tolerance applies to at one position."""
    return max(1.0, max(abs(v) for v in values))


def tie_set(top: Dict[str, Any], logit_tolerance: float) -> Set[int]:
    """The token IDs within logit tolerance of the top token."""
    if not top or not top.get("values"):
        return set()
    best = top["values"][0]
    allowed = logit_tolerance * _scale(top["values"])
    return {i for i, v in zip(top["ids"], top["values"]) if best - v <= allowed}


def _first_flip(r: Dict[str, Any], p: Dict[str, Any], logit_tolerance: float) -> Optional[Dict[str, Any]]:
    """Determine first position where generated tokens diverge."""
    r_gen = r.get("generated_ids", [])
    p_gen = p.get("generated_ids", [])
    pos = next((i for i, (x, y) in enumerate(zip(r_gen, p_gen)) if x != y), None)
    if pos is None:
        if len(r_gen) == len(p_gen):
            return None
        pos = min(len(r_gen), len(p_gen))

    flip: Dict[str, Any] = {
        "position": pos,
        "ref_token": r_gen[pos] if pos < len(r_gen) else None,
        "token": p_gen[pos] if pos < len(p_gen) else None,
    }
    if flip["token"] is None or flip["ref_token"] is None:
        flip["kind"] = "length"
        return flip

    tops = r.get("top_logits") or []
    if pos >= len(tops) or not tops[pos] or not tops[pos].get("values"):
        flip["kind"] = "unjudged"
        return flip

    top = tops[pos]
    values = dict(zip(top["ids"], top["values"]))
    best = top["values"][0]
    flip["gap"] = round(best - values[flip["token"]], 6) if flip["token"] in values else None
    flip["kind"] = "tie" if flip["token"] in tie_set(top, logit_tolerance) else "outside"
    return flip


def compare_vectors(
    reference: Dict[str, Any], candidate: Dict[str, Any], logit_tolerance: float = 0.025
) -> Tuple[List[Dict[str, Any]], bool]:
    """Compare candidate vectors against reference vectors."""
    rows = []
    ok = True
    ref = {p["name"]: p for p in reference.get("prompts", [])}

    for p in candidate.get("prompts", []):
        r = ref.get(p["name"])
        if r is None:
            rows.append({
                "prompt": p["name"],
                "status": "no reference",
                "compared": "0/0",
                "first_flip_pos": None,
                "flip": None,
                "max_logit_diff": None,
                "max_logit_rel": None,
                "ref_ms_per_token": None,
                "ms_per_token": p.get("ms_per_token"),
            })
            continue

        same_prompt = r.get("input_ids") == p.get("input_ids")
        flip = _first_flip(r, p, logit_tolerance)
        tokens_equal = flip is None
        max_diff, max_rel = None, None
        r_gen_len = len(r.get("generated_ids", []))
        comparable = r_gen_len if flip is None else min(flip["position"] + 1, r_gen_len)

        if r.get("top_logits") and p.get("top_logits"):
            diffs: List[float] = []
            rels: List[float] = []
            for pos, (a, b) in enumerate(list(zip(r["top_logits"], p["top_logits"]))[:comparable]):
                if not a or not b or not a.get("values") or not b.get("values"):
                    continue
                av = dict(zip(a["ids"], a["values"]))
                bv = dict(zip(b["ids"], b["values"]))
                a_floor = min(a["values"])
                b_floor = min(b["values"])
                scale = _scale(a["values"])
                here = []
                for i, v in av.items():
                    here.append(abs(v - bv[i]) if i in bv else max(0.0, v - b_floor))
                for i, v in bv.items():
                    if i not in av:
                        here.append(max(0.0, v - a_floor))
                diffs.extend(here)
                rels.extend(d / scale for d in here)
            max_diff = max(diffs) if diffs else None
            max_rel = max(rels) if rels else None

        tokens_ok = tokens_equal or (flip is not None and flip.get("kind") == "tie")
        row_ok = same_prompt and tokens_ok and (max_rel is None or max_rel <= logit_tolerance)
        ok &= row_ok

        rows.append({
            "prompt": p["name"],
            "status": "ok" if row_ok else "DIFF",
            "same_prompt": same_prompt,
            "tokens_equal": tokens_equal,
            "tokens_ok": tokens_ok,
            "flip": flip,
            "first_flip_pos": None if flip is None else flip.get("position"),
            "compared": f"{comparable}/{r_gen_len}",
            "max_logit_diff": None if max_diff is None else round(max_diff, 4),
            "max_logit_rel": None if max_rel is None else round(max_rel, 4),
            "ref_ms_per_token": r.get("ms_per_token"),
            "ms_per_token": p.get("ms_per_token"),
        })

    return rows, ok


# --- Discovery & Parsing ---


def _load_json(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load JSON from {file_path}: {e}", file=sys.stderr)
        return None


def _is_attribute_decode(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return False
    keys = {"clean_ms_per_token", "per_step_ms", "dense_segments", "decode_steps"}
    return any(k in doc for k in keys)


def _is_vectors(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("format") == "vxpu-vectors/1"


def _is_compare(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return False
    if doc.get("format") == "vxpu-vectors-compare/1":
        return True
    if "rows" in doc and isinstance(doc["rows"], list) and doc["rows"]:
        first = doc["rows"][0]
        if isinstance(first, dict) and ("compared" in first or "tokens_compared" in first or "flip" in first):
            return True
    return False


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", text).strip("-")
    return slug or "run"


def find_runs(results_dir: str, default_reference_path: Optional[str] = None) -> List[Run]:
    """Recursively search results_dir and parse runs."""
    results_dir = os.path.abspath(results_dir)
    discovered_dirs: List[str] = []

    # Find all directories that contain at least one json file
    for root, _, files in os.walk(results_dir):
        if any(f.endswith(".json") for f in files):
            discovered_dirs.append(root)

    # Sort directories by path depth (deepest first or top-first)
    discovered_dirs.sort()

    # Identify potential root/global reference files
    global_ref_doc: Optional[Dict[str, Any]] = None
    global_ref_name: Optional[str] = None

    if default_reference_path and os.path.isfile(default_reference_path):
        global_ref_doc = _load_json(default_reference_path)
        global_ref_name = os.path.basename(default_reference_path)
    else:
        # Check if root results_dir has a reference file
        for name in os.listdir(results_dir):
            if name.endswith(".json") and ("ref" in name.lower() or "reference" in name.lower()):
                candidate_path = os.path.join(results_dir, name)
                doc = _load_json(candidate_path)
                if _is_vectors(doc):
                    global_ref_doc = doc
                    global_ref_name = name
                    break

    runs: List[Run] = []

    for d in discovered_dirs:
        json_files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")]
        if not json_files:
            continue

        run_meta: Dict[str, Any] = {}
        attr_data: Optional[Dict[str, Any]] = None
        vector_candidates: List[Tuple[str, Dict[str, Any]]] = []
        compare_data: Optional[Dict[str, Any]] = None
        local_ref_doc: Optional[Dict[str, Any]] = None
        local_ref_name: Optional[str] = None

        for path in json_files:
            fname = os.path.basename(path)
            doc = _load_json(path)
            if doc is None:
                continue

            if fname == "run.json":
                if isinstance(doc, dict):
                    run_meta = doc
            elif _is_compare(doc):
                compare_data = doc
            elif _is_attribute_decode(doc):
                attr_data = doc
            elif _is_vectors(doc):
                producer_kind = (doc.get("producer") or {}).get("kind", "")
                if "ref" in fname.lower() or "reference" in fname.lower() or producer_kind == "transformers":
                    local_ref_doc = doc
                    local_ref_name = fname
                else:
                    vector_candidates.append((fname, doc))

        # Check if this directory is only a container for a shared reference and has subdirectories
        has_child_runs = any(
            sub != d and sub.startswith(d + os.sep) for sub in discovered_dirs
        )
        if has_child_runs and not run_meta and not attr_data and not vector_candidates and not compare_data:
            # Container directory, skip creating a run for it
            continue

        # If there's neither run_meta, attr_data, vector_candidates, nor compare_data, skip
        if not run_meta and not attr_data and not vector_candidates and not compare_data:
            continue

        rel_path = os.path.relpath(d, results_dir)
        run_slug = _slugify(rel_path if rel_path != "." else os.path.basename(d))
        label = run_meta.get("label") or (os.path.basename(d) if rel_path != "." else "Run")

        # Fallback date to dir modification time if not provided
        date_str = run_meta.get("date")
        if not date_str:
            try:
                mtime = os.path.getmtime(d)
                import datetime
                date_str = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                date_str = None

        commit = run_meta.get("commit")
        hardware = run_meta.get("hardware")
        artifact = run_meta.get("artifact")
        notes = run_meta.get("notes")

        # If hardware or commit not in run_meta, check vector candidate metadata
        if not hardware and vector_candidates:
            producer = vector_candidates[0][1].get("producer") or {}
            hardware = producer.get("gpu") or producer.get("device")
        if not commit and vector_candidates:
            commit = (vector_candidates[0][1].get("model") or {}).get("commit_sha")

        run = Run(
            id=run_slug,
            dir_path=d,
            label=label,
            commit=commit,
            artifact=artifact,
            hardware=hardware,
            date=date_str,
            notes=notes,
        )

        # Populate attribute_decode data
        if attr_data:
            run.clean_ms_per_token = attr_data.get("clean_ms_per_token")
            run.clean_best_ms_per_token = attr_data.get("clean_best_ms_per_token")
            run.clean_sessions_ms_per_token = attr_data.get("clean_sessions_ms_per_token")
            run.decode_steps = attr_data.get("decode_steps")
            run.instrumented_ms_per_token = attr_data.get("instrumented_ms_per_token")
            run.per_step_ms = attr_data.get("per_step_ms") or {}
            run.experts_routing_body_ms = attr_data.get("experts_routing_body_ms")
            run.dense_segments = attr_data.get("dense_segments") or []

        # Populate vectors comparison data
        if compare_data:
            rows = compare_data.get("rows")
            if isinstance(rows, list):
                # Standardize rows
                standardized = []
                for r in rows:
                    std_row = dict(r)
                    if "tokens_compared" in std_row and "compared" not in std_row:
                        std_row["compared"] = std_row["tokens_compared"]
                    if "first_flip_position" in std_row and "first_flip_pos" not in std_row:
                        std_row["first_flip_pos"] = std_row["first_flip_position"]
                    elif "flip" in std_row and isinstance(std_row["flip"], dict) and "first_flip_pos" not in std_row:
                        std_row["first_flip_pos"] = std_row["flip"].get("position")
                    standardized.append(std_row)
                run.vectors_rows = standardized
            run.vectors_ok = compare_data.get("ok", True)
            run.vectors_reference_name = compare_data.get("reference")
        elif vector_candidates:
            cand_name, cand_doc = vector_candidates[0]
            run.raw_vectors = cand_doc

            # Determine reference
            ref_doc = None
            ref_name = None

            # 1. run.json reference field
            if run_meta.get("reference"):
                ref_path = os.path.normpath(os.path.join(d, run_meta["reference"]))
                if os.path.isfile(ref_path):
                    ref_doc = _load_json(ref_path)
                    ref_name = os.path.basename(ref_path)

            # 2. Local reference in same directory
            if ref_doc is None and local_ref_doc:
                ref_doc = local_ref_doc
                ref_name = local_ref_name

            # 3. Global reference
            if ref_doc is None and global_ref_doc:
                ref_doc = global_ref_doc
                ref_name = global_ref_name

            if ref_doc is not None:
                rows, ok = compare_vectors(ref_doc, cand_doc)
                run.vectors_rows = rows
                run.vectors_ok = ok
                run.vectors_reference_name = ref_name

        runs.append(run)

    # Sort runs chronologically if dates exist, or keep discovery order
    def sort_key(r: Run) -> str:
        return r.date or r.label

    runs.sort(key=sort_key)
    return runs


# --- SVG Charts & Bars ---


def render_svg_chart(runs: List[Run]) -> str:
    """Generate an inline SVG chart of ms/token over runs without JS libraries."""
    valid_runs = [r for r in runs if r.clean_ms_per_token is not None or r.clean_best_ms_per_token is not None]
    if not valid_runs:
        return """
        <div class="empty-chart">
            <p>No latency measurements (ms/token) available to chart.</p>
        </div>
        """

    width = 850
    height = 320
    ml = 65
    mr = 40
    mt = 40
    mb = 75

    pw = width - ml - mr
    ph = height - mt - mb

    all_vals: List[float] = []
    for r in valid_runs:
        if r.clean_ms_per_token is not None:
            all_vals.append(r.clean_ms_per_token)
        if r.clean_best_ms_per_token is not None:
            all_vals.append(r.clean_best_ms_per_token)

    max_v = max(all_vals)
    y_max = math.ceil(max_v * 1.15 / 5.0) * 5.0
    if y_max <= 0:
        y_max = 10.0
    y_min = 0.0

    def val_to_y(v: float) -> float:
        norm = (v - y_min) / (y_max - y_min)
        return (mt + ph) - norm * ph

    # Y ticks (5 ticks)
    ticks = 5
    y_ticks = [y_min + (y_max - y_min) * (i / ticks) for i in range(ticks + 1)]

    svg_parts = [
        f'<svg class="latency-chart" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '  <defs>',
        '    <linearGradient id="median-area-grad" x1="0%" y1="0%" x2="0%" y2="100%">',
        '      <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25" />',
        '      <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0" />',
        '    </linearGradient>',
        '  </defs>',
    ]

    # Grid lines & labels
    for tick_val in y_ticks:
        y_pos = val_to_y(tick_val)
        svg_parts.append(
            f'  <line x1="{ml}" y1="{y_pos:.1f}" x2="{width - mr}" y2="{y_pos:.1f}" '
            f'class="chart-grid" stroke-dasharray="3 3" />'
        )
        svg_parts.append(
            f'  <text x="{ml - 10}" y="{y_pos + 4:.1f}" class="chart-axis-text" text-anchor="end">'
            f'{tick_val:.1f} ms</text>'
        )

    # Compute X coordinates for runs
    n = len(valid_runs)
    run_xs: List[float] = []
    for i in range(n):
        if n == 1:
            run_xs.append(ml + pw / 2)
        else:
            run_xs.append(ml + i * (pw / (n - 1)))

    # Median points & Best points
    median_coords: List[Tuple[float, float, Run]] = []
    best_coords: List[Tuple[float, float, Run]] = []

    for idx, r in enumerate(valid_runs):
        rx = run_xs[idx]
        if r.clean_ms_per_token is not None:
            median_coords.append((rx, val_to_y(r.clean_ms_per_token), r))
        if r.clean_best_ms_per_token is not None:
            best_coords.append((rx, val_to_y(r.clean_best_ms_per_token), r))

    # Draw Median Area Fill
    if len(median_coords) > 1:
        d_area_pts = [f"{x:.1f},{y:.1f}" for x, y, _ in median_coords]
        last_x = median_coords[-1][0]
        first_x = median_coords[0][0]
        bottom_y = mt + ph
        d_area = f"M {d_area_pts[0]} " + " ".join(f"L {p}" for p in d_area_pts[1:])
        d_area += f" L {last_x:.1f},{bottom_y} L {first_x:.1f},{bottom_y} Z"
        svg_parts.append(f'  <path d="{d_area}" fill="url(#median-area-grad)" />')

    # Draw Best Polyline
    if len(best_coords) > 1:
        d_best = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y, _ in best_coords)
        svg_parts.append(
            f'  <path d="{d_best}" fill="none" stroke="#10b981" stroke-width="2" '
            f'stroke-dasharray="4 4" stroke-linecap="round" />'
        )

    # Draw Median Polyline
    if len(median_coords) > 1:
        d_med = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y, _ in median_coords)
        svg_parts.append(
            f'  <path d="{d_med}" fill="none" stroke="#3b82f6" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round" />'
        )

    # Draw Best Points
    for x, y, r in best_coords:
        b_val = r.clean_best_ms_per_token or 0.0
        svg_parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#10b981" stroke="#ffffff" stroke-width="1.5" class="chart-point">'
            f'<title>{html.escape(r.label)}: {b_val:.1f} ms/token (best)</title></circle>'
        )

    # Draw Median Points & Labels
    for x, y, r in median_coords:
        m_val = r.clean_ms_per_token or 0.0
        svg_parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#3b82f6" stroke="#ffffff" stroke-width="2" class="chart-point">'
            f'<title>{html.escape(r.label)}: {m_val:.1f} ms/token (median)</title></circle>'
        )
        if n <= 10:
            svg_parts.append(
                f'  <text x="{x:.1f}" y="{y - 10:.1f}" class="chart-point-val" text-anchor="middle">{m_val:.1f}</text>'
            )

    # X-axis Labels
    for idx, r in enumerate(valid_runs):
        rx = run_xs[idx]
        display_label = r.label if len(r.label) <= 16 else r.label[:15] + "…"
        svg_parts.append(
            f'  <text x="{rx:.1f}" y="{mt + ph + 22:.1f}" class="chart-x-label" '
            f'text-anchor="end" transform="rotate(-35 {rx:.1f} {mt + ph + 22:.1f})">'
            f'<title>{html.escape(r.label)}</title>{html.escape(display_label)}</text>'
        )

    # Legend at top right
    leg_x = width - mr - 230
    leg_y = mt - 15
    svg_parts.extend([
        f'  <g class="chart-legend" transform="translate({leg_x}, {leg_y})">',
        '    <circle cx="10" cy="5" r="4.5" fill="#3b82f6" />',
        '    <line x1="2" y1="5" x2="18" y2="5" stroke="#3b82f6" stroke-width="2" />',
        '    <text x="24" y="9" class="chart-legend-text">Median ms/token</text>',
        '    <circle cx="130" cy="5" r="3.5" fill="#10b981" />',
        '    <line x1="122" y1="5" x2="138" y2="5" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3 3" />',
        '    <text x="144" y="9" class="chart-legend-text">Best ms/token</text>',
        '  </g>',
        '</svg>',
    ])

    return "\n".join(svg_parts)


def render_stacked_bar(per_step_ms: Dict[str, float]) -> str:
    """Generate an inline SVG horizontal stacked bar for per-step time by kind."""
    if not per_step_ms:
        return "<p class='text-muted'>No per-step breakdown recorded.</p>"

    total_ms = sum(per_step_ms.values())
    if total_ms <= 0:
        return "<p class='text-muted'>Invalid step timings.</p>"

    palette = {
        "dense": "#3b82f6",
        "experts": "#8b5cf6",
        "engram": "#f59e0b",
        "attention": "#06b6d4",
        "mlp": "#ec4899",
    }
    fallback_colors = ["#10b981", "#6366f1", "#f97316", "#14b8a6", "#84cc16"]

    width = 750
    height = 42
    bar_h = 32

    svg_parts = [
        f'<svg class="step-bar" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '  <defs>',
        f'    <clipPath id="bar-clip"><rect x="0" y="5" width="{width}" height="{bar_h}" rx="6" ry="6" /></clipPath>',
        '  </defs>',
        '  <g clip-path="url(#bar-clip)">',
    ]

    curr_x = 0.0
    color_idx = 0
    legend_items = []

    for kind, ms in per_step_ms.items():
        pct = (ms / total_ms) * 100.0
        seg_w = (ms / total_ms) * width
        color = palette.get(kind.lower())
        if not color:
            color = fallback_colors[color_idx % len(fallback_colors)]
            color_idx += 1

        label = f"{kind}: {ms:.2f} ms ({pct:.1f}%)"
        svg_parts.append(
            f'    <rect x="{curr_x:.1f}" y="5" width="{seg_w:.1f}" height="{bar_h}" fill="{color}">'
            f'<title>{html.escape(label)}</title></rect>'
        )

        # Label inside if segment is wide enough
        if seg_w > 65:
            mid_x = curr_x + seg_w / 2
            svg_parts.append(
                f'    <text x="{mid_x:.1f}" y="25" text-anchor="middle" fill="#ffffff" '
                f'font-size="12" font-weight="600">{kind} ({pct:.0f}%)</text>'
            )

        legend_items.append((kind, ms, pct, color))
        curr_x += seg_w

    svg_parts.append('  </g>')
    svg_parts.append('</svg>')

    # Legend below bar
    legend_html = ['<div class="bar-legend">']
    for kind, ms, pct, color in legend_items:
        legend_html.append(
            f'<div class="bar-legend-item">'
            f'<span class="legend-color-dot" style="background-color: {color};"></span>'
            f'<span class="legend-label"><strong>{html.escape(kind)}</strong>: {ms:.2f} ms ({pct:.1f}%)</span>'
            f'</div>'
        )
    legend_html.append('</div>')

    return "\n".join(svg_parts) + "\n" + "\n".join(legend_html)


# --- Common Styles & HTML Template ---


def _common_styles() -> str:
    return """
:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --text: #0f172a;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --success: #16a34a;
  --success-bg: #dcfce7;
  --warning: #d97706;
  --warning-bg: #fef3c7;
  --danger: #dc2626;
  --danger-bg: #fee2e2;
  --radius: 8px;
  --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --surface: #1e293b;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
    --primary: #3b82f6;
    --primary-hover: #60a5fa;
    --success: #22c55e;
    --success-bg: #064e3b;
    --warning: #f59e0b;
    --warning-bg: #78350f;
    --danger: #ef4444;
    --danger-bg: #7f1d1d;
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  padding: 24px;
}

.container {
  max-width: 1100px;
  margin: 0 auto;
}

header {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}

.header-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}

h1 { font-size: 24px; font-weight: 700; }
h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; }

.subtitle {
  color: var(--text-muted);
  font-size: 14px;
  margin-top: 4px;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.grid-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.stat-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

.stat-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--text-muted);
  letter-spacing: 0.05em;
  margin-bottom: 6px;
}

.stat-value {
  font-size: 22px;
  font-weight: 700;
  font-family: var(--font-mono);
}

.stat-sub {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 4px;
}

/* Tables */
.table-responsive {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  text-align: left;
}

th, td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}

th {
  background: rgba(148, 163, 184, 0.08);
  font-weight: 600;
  font-size: 13px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

tr:hover td {
  background: rgba(148, 163, 184, 0.05);
}

td.mono { font-family: var(--font-mono); font-size: 13px; }

/* Badges */
.badge {
  display: inline-block;
  padding: 3px 8px;
  font-size: 12px;
  font-weight: 600;
  border-radius: 4px;
  white-space: nowrap;
}

.badge-success { background: var(--success-bg); color: var(--success); }
.badge-warning { background: var(--warning-bg); color: var(--warning); }
.badge-danger { background: var(--danger-bg); color: var(--danger); }
.badge-neutral { background: rgba(148, 163, 184, 0.15); color: var(--text-muted); }
.badge-mono { font-family: var(--font-mono); font-size: 11px; }

/* Links */
a { color: var(--primary); text-decoration: none; }
a:hover { text-decoration: underline; color: var(--primary-hover); }

.back-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 16px;
}

/* Metadata List */
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  font-size: 13px;
}

.meta-item strong {
  color: var(--text-muted);
  display: inline-block;
  min-width: 80px;
}

/* SVG Chart */
.latency-chart {
  width: 100%;
  height: auto;
  display: block;
}

.chart-grid {
  stroke: var(--border);
  stroke-width: 1;
}

.chart-axis-text {
  font-size: 11px;
  fill: var(--text-muted);
  font-family: var(--font-mono);
}

.chart-x-label {
  font-size: 11px;
  fill: var(--text-muted);
  font-family: var(--font-sans);
}

.chart-point-val {
  font-size: 11px;
  font-weight: 600;
  fill: var(--primary);
  font-family: var(--font-mono);
}

.chart-legend-text {
  font-size: 12px;
  fill: var(--text);
  font-weight: 500;
}

.empty-chart {
  padding: 40px;
  text-align: center;
  color: var(--text-muted);
}

/* Stacked bar */
.step-bar {
  width: 100%;
  height: auto;
  display: block;
}

.bar-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 12px;
  font-size: 13px;
}

.bar-legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend-color-dot {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  display: inline-block;
}

.callout {
  padding: 12px 16px;
  border-left: 4px solid var(--primary);
  background: rgba(59, 130, 246, 0.08);
  border-radius: 0 var(--radius) var(--radius) 0;
  margin-top: 16px;
  font-size: 13px;
}

.op-tag {
  display: inline-block;
  background: rgba(148, 163, 184, 0.12);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: 11px;
  margin: 2px;
}
"""


# --- HTML Page Renderers ---


def render_index_page(runs: List[Run], title: str = "vXPU Performance & Verification Results") -> str:
    """Generate index.html listing runs and an inline SVG chart."""
    svg_chart = render_svg_chart(runs)

    # Table rows
    table_rows = []
    for r in runs:
        run_link = f'<a href="run-{html.escape(r.id)}.html"><strong>{html.escape(r.label)}</strong></a>'
        date_display = html.escape(r.date.split("T")[0] if r.date and "T" in r.date else (r.date or "-"))
        hw_display = html.escape(r.hardware or "-")
        commit_display = f'<span class="badge badge-neutral badge-mono">{html.escape(r.commit[:7])}</span>' if r.commit else "-"

        med_ms = f"{r.clean_ms_per_token:.1f} ms" if r.clean_ms_per_token is not None else "-"
        best_ms = f"{r.clean_best_ms_per_token:.1f} ms" if r.clean_best_ms_per_token is not None else "-"

        # Status / vectors summary
        if r.vectors_rows:
            if r.vectors_ok:
                diff_count = sum(1 for row in r.vectors_rows if row.get("flip"))
                if diff_count == 0:
                    status_badge = '<span class="badge badge-success">Equal</span>'
                else:
                    status_badge = f'<span class="badge badge-warning">Tie flip ({diff_count})</span>'
            else:
                status_badge = '<span class="badge badge-danger">DIFF</span>'
        else:
            status_badge = '<span class="badge badge-neutral">No vectors</span>'

        notes_display = html.escape(r.notes or "")

        table_rows.append(f"""
        <tr>
          <td>{run_link}</td>
          <td>{date_display}</td>
          <td>{hw_display}</td>
          <td>{commit_display}</td>
          <td class="mono"><strong>{med_ms}</strong></td>
          <td class="mono">{best_ms}</td>
          <td>{status_badge}</td>
          <td><small class="text-muted">{notes_display}</small></td>
        </tr>
        """)

    # Overall summary numbers
    total_runs = len(runs)
    measured_runs = [r for r in runs if r.clean_best_ms_per_token is not None]
    best_overall = min((r.clean_best_ms_per_token for r in measured_runs), default=None)
    best_overall_str = f"{best_overall:.1f} ms/token" if best_overall is not None else "-"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
{_common_styles()}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div>
          <h1>{html.escape(title)}</h1>
          <p class="subtitle">Decode latency trajectory and test-vector equivalence across hardware and transformations</p>
        </div>
      </div>
    </header>

    <div class="grid-stats">
      <div class="stat-box">
        <div class="stat-label">Total Runs</div>
        <div class="stat-value">{total_runs}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Best Decode Rate</div>
        <div class="stat-value" style="color: var(--success);">{best_overall_str}</div>
        <div class="stat-sub">Across all recorded sessions</div>
      </div>
    </div>

    <div class="card">
      <h2>Latency Trajectory (ms/token over runs)</h2>
      {svg_chart}
    </div>

    <div class="card">
      <h2>Runs Overview</h2>
      <div class="table-responsive">
        <table>
          <thead>
            <tr>
              <th>Label</th>
              <th>Date</th>
              <th>Hardware</th>
              <th>Commit</th>
              <th>Median</th>
              <th>Best</th>
              <th>Vectors</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>
"""


def render_run_page(run: Run, title_prefix: str = "vXPU Run") -> str:
    """Generate run-<id>.html for an individual benchmark run."""
    # Stacked bar
    step_bar_html = render_stacked_bar(run.per_step_ms)

    # Experts routing body callout
    expert_callout = ""
    if run.experts_routing_body_ms is not None and "experts" in run.per_step_ms:
        exp_total = run.per_step_ms["experts"]
        body_ms = run.experts_routing_body_ms
        dispatch_ms = max(0.0, exp_total - body_ms)
        body_pct = (body_ms / exp_total) * 100.0 if exp_total > 0 else 0.0
        expert_callout = f"""
        <div class="callout">
          <strong>Inside the Expert Host Op:</strong> routing body (CPU kernels &amp; glue) took
          <strong>{body_ms:.1f} ms</strong> ({body_pct:.1f}% of {exp_total:.1f} ms in the op);
          device crossings and dispatch took <strong>{dispatch_ms:.1f} ms</strong>.
        </div>
        """

    # Heaviest segments table
    dense_table_rows = []
    total_dense_ms = run.per_step_ms.get("dense", sum(s.get("ms", 0.0) for s in run.dense_segments))
    for seg in run.dense_segments:
        s_name = html.escape(seg.get("segment", "unknown"))
        s_ms = seg.get("ms", 0.0)
        s_pct = (s_ms / total_dense_ms * 100.0) if total_dense_ms > 0 else 0.0
        s_nodes = seg.get("nodes", 0)

        # Format ops
        ops_html = []
        for op, cnt in seg.get("ops", []):
            ops_html.append(f'<span class="op-tag">{html.escape(str(op))}: {cnt}</span>')
        ops_str = " ".join(ops_html) if ops_html else "-"

        # Format modules
        mods_html = []
        for mod, cnt in seg.get("modules", []):
            mods_html.append(f'<span class="op-tag">{html.escape(str(mod))} ({cnt})</span>')
        mods_str = " ".join(mods_html) if mods_html else "-"

        dense_table_rows.append(f"""
        <tr>
          <td><span class="badge badge-mono">{s_name}</span></td>
          <td class="mono"><strong>{s_ms:.2f} ms</strong></td>
          <td class="mono">{s_pct:.1f}%</td>
          <td class="mono">{s_nodes}</td>
          <td>{ops_str}</td>
          <td>{mods_str}</td>
        </tr>
        """)

    if dense_table_rows:
        segments_card_content = f"""
        <div class="table-responsive">
          <table>
            <thead>
              <tr>
                <th>Segment</th>
                <th>Time (ms)</th>
                <th>% of Dense</th>
                <th>Nodes</th>
                <th>Frequent Ops</th>
                <th>Source Modules</th>
              </tr>
            </thead>
            <tbody>
              {''.join(dense_table_rows)}
            </tbody>
          </table>
        </div>
        """
    else:
        segments_card_content = "<p class='text-muted'>No dense segment attribution recorded for this run.</p>"

    # Vectors table
    vector_table_rows = []
    for v in run.vectors_rows:
        prompt_name = html.escape(str(v.get("prompt", "-")))
        compared = html.escape(str(v.get("compared", "-")))

        flip = v.get("flip")
        if flip is None:
            flip_display = '<span class="badge badge-success">Equal (no flip)</span>'
        else:
            f_pos = flip.get("position", v.get("first_flip_pos", "-"))
            f_kind = flip.get("kind", "tie")
            f_gap = flip.get("gap")
            gap_info = f", gap {f_gap}" if f_gap is not None else ""
            badge_cls = "badge-warning" if f_kind == "tie" else "badge-danger"
            flip_display = f'<span class="badge {badge_cls}">Flip @ {f_pos} ({f_kind}{gap_info})</span>'

        max_diff = v.get("max_logit_diff")
        max_rel = v.get("max_logit_rel")
        if max_diff is not None:
            rel_str = f" ({max_rel * 100:.2f}%)" if max_rel is not None else ""
            diff_display = f"{max_diff:.4f}{rel_str}"
        else:
            diff_display = "-"

        v_status = v.get("status", "ok")
        status_badge = '<span class="badge badge-success">OK</span>' if v_status == "ok" else '<span class="badge badge-danger">DIFF</span>'

        ms_ref = v.get("ref_ms_per_token")
        ms_cand = v.get("ms_per_token")
        rate_display = f"{ms_ref:.1f} → {ms_cand:.1f}" if ms_ref and ms_cand else (f"{ms_cand:.1f}" if ms_cand else "-")

        vector_table_rows.append(f"""
        <tr>
          <td><strong>{prompt_name}</strong></td>
          <td class="mono">{compared}</td>
          <td>{flip_display}</td>
          <td class="mono">{diff_display}</td>
          <td class="mono">{rate_display}</td>
          <td>{status_badge}</td>
        </tr>
        """)

    ref_name_str = html.escape(run.vectors_reference_name or "default reference")
    if vector_table_rows:
        vectors_card_content = f"""
        <p class="subtitle" style="margin-bottom: 12px;">Reference: <code>{ref_name_str}</code></p>
        <div class="table-responsive">
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>Tokens Compared</th>
                <th>First Flip Position</th>
                <th>Max Logit Diff (abs / rel)</th>
                <th>ms/token (ref → cand)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {''.join(vector_table_rows)}
            </tbody>
          </table>
        </div>
        """
    else:
        vectors_card_content = "<p class='text-muted'>No test vectors recorded or compared for this run.</p>"

    # Clean sessions string
    clean_sessions_str = ""
    if run.clean_sessions_ms_per_token:
        clean_sessions_str = ", ".join(f"{s:.1f}" for s in run.clean_sessions_ms_per_token)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(run.label)} - {html.escape(title_prefix)}</title>
  <style>
{_common_styles()}
  </style>
</head>
<body>
  <div class="container">
    <a href="index.html" class="back-link">← Back to Overview</a>

    <header>
      <h1>{html.escape(run.label)}</h1>
      <p class="subtitle">{html.escape(run.notes or "Benchmark and verification run details")}</p>
    </header>

    <div class="grid-stats">
      <div class="stat-box">
        <div class="stat-label">Clean Median ms/token</div>
        <div class="stat-value">{f"{run.clean_ms_per_token:.1f} ms" if run.clean_ms_per_token else "-"}</div>
        <div class="stat-sub">{f"Best: {run.clean_best_ms_per_token:.1f} ms" if run.clean_best_ms_per_token else ""}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Decode Steps</div>
        <div class="stat-value">{run.decode_steps or "-"}</div>
        <div class="stat-sub">{f"Instrumented: {run.instrumented_ms_per_token:.1f} ms" if run.instrumented_ms_per_token else ""}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Hardware</div>
        <div class="stat-value" style="font-size: 16px;">{html.escape(run.hardware or "-")}</div>
        <div class="stat-sub">{f"Commit: {run.commit[:7]}" if run.commit else ""}</div>
      </div>
    </div>

    <div class="card">
      <h2>Metadata</h2>
      <div class="meta-grid">
        <div class="meta-item"><strong>Commit:</strong> <span class="badge badge-mono">{html.escape(run.commit or "-")}</span></div>
        <div class="meta-item"><strong>Date:</strong> {html.escape(run.date or "-")}</div>
        <div class="meta-item"><strong>Hardware:</strong> {html.escape(run.hardware or "-")}</div>
        <div class="meta-item"><strong>Artifact:</strong> {html.escape(run.artifact or "-")}</div>
        {f'<div class="meta-item" style="grid-column: 1 / -1;"><strong>Sessions:</strong> <span class="mono">[{html.escape(clean_sessions_str)}] ms/token</span></div>' if clean_sessions_str else ''}
      </div>
    </div>

    <div class="card">
      <h2>Per-Step Decode Time by Kind</h2>
      {step_bar_html}
      {expert_callout}
    </div>

    <div class="card">
      <h2>Heaviest Segments</h2>
      {segments_card_content}
    </div>

    <div class="card">
      <h2>Test Vectors Verification</h2>
      {vectors_card_content}
    </div>
  </div>
</body>
</html>
"""


# --- Main Build Orchestration ---


def build_site(results_dir: str, out_dir: str, default_reference_path: Optional[str] = None, title: str = "vXPU Performance & Verification Results") -> None:
    """Build the static results site from results_dir into out_dir."""
    runs = find_runs(results_dir, default_reference_path)
    os.makedirs(out_dir, exist_ok=True)

    # Render index.html
    index_html = render_index_page(runs, title=title)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    # Render individual run pages
    for run in runs:
        run_html = render_run_page(run)
        with open(os.path.join(out_dir, f"run-{run.id}.html"), "w", encoding="utf-8") as f:
            f.write(run_html)

    print(f"Successfully generated static results site in {out_dir} ({len(runs)} runs).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", help="Directory containing run results (searched recursively)")
    parser.add_argument("out_dir", help="Directory where static HTML files will be written")
    parser.add_argument("--reference", help="Reference vectors file (vxpu-vectors/1) to compare against", default=None)
    parser.add_argument("--title", help="Site title", default="vXPU Performance & Verification Results")
    args = parser.parse_args()

    build_site(args.results_dir, args.out_dir, default_reference_path=args.reference, title=args.title)


if __name__ == "__main__":
    main()
