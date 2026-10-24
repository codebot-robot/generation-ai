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

"""Replay recorded traffic against expert placement policies, offline.

A placement holds some experts on the GPU (experts_gpu.py); the rest are
computed on the CPU, and the CPU's share of the selections sets the
decode step's expert time. Whether the placement should follow the
traffic (move experts as the hot set drifts) or stay fixed is a question
the engine's own records can answer before any of it is built: replay
the requests in time order and ask each policy, before each request,
which experts it would hold.

Subcommands:
  requests   replay request-level summary records against placement policies
  steps      replay per-step ordered selection traces through cache policies,
             storage tiers, and prefetch predictors (#348, #447)

Policies:
  static       the hottest experts of a training prefix of the requests or
               steps, held for the rest (what VXPU_GPU_EXPERTS does today)
  adaptive     decayed LFU between requests: every expert's score decays
               after each request and its selections are added.
  decayed_lfu  decayed LFU per decode step: score decays every step. With
               decay 1.0 this is cumulative LFU.
  lru          least recently used eviction per layer (steps mode)
  prefill      the request's own prefill mixed with cumulative history
  oracle       in requests mode: each request's own hottest decode experts,
               chosen in hindsight. This is an upper bound for any policy that
               places once per request. In steps mode: best static
               placement in hindsight across the evaluated steps. A replacement
               policy such as LRU can beat it.

Usage:
    python -m vxpu.expert_sim requests RECORDS... [--artifact DIR | --expert-mb 18.8] [--budgets-gb 60,165]
    python -m vxpu.expert_sim steps TRACES... [--artifact DIR | --expert-mb 18.8] [--budgets-gb 60,165]
"""

import argparse
import collections
import json
import math
import statistics

from . import expert_stats

CPU_FIXED_MS = 3.5  # per decode step: the op's 40 calls, the copies, the threads' start
CPU_PAIR_MS = 0.076  # per (token, expert) pair on the CPU kernels
DECAYS = (1.0, 0.9, 0.7, 0.5, 0.0)
PREFILL_WEIGHTS = (0.25, 0.5, 0.75)


def requests(docs):
    """[(counts, steps, prefill)] in time order: counts a Counter of
    (layer, expert) -> decode selections, steps the decode steps they
    cover, prefill the same request's prefill counts (what the prompt says
    about its decode; the policies and the CPU model are decode's).
    Request records (vxpu-request/1) are sorted by their time; documents
    without one keep the order they were read in."""
    out = []
    for i, doc in enumerate(docs):
        for tel in expert_stats._records(doc):
            counts, prefill = _counts(tel, "decode"), _counts(tel, "prefill")
            if counts:
                out.append((doc.get("time") or "", i, counts, tel.get("decode_steps") or 0, prefill))
    out.sort(key=lambda r: (r[0], r[1]))
    return [(counts, steps, prefill) for _, _, counts, steps, prefill in out]


def _counts(tel, phase):
    counts = collections.Counter()
    for layer, c in ((tel.get("experts") or {}).get(phase) or {}).items():
        for expert, n in c.items():
            counts[(int(layer), int(expert))] += n
    return counts


def top(scores, k):
    """The k highest-scoring experts (ties by layer, then expert)."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {e for e, s in ranked[:k] if s > 0}


def simulate(reqs, k, train, policy, decay=1.0, prefill_weight=0.5):
    """Per evaluated request: (share, moved). reqs as from requests(),
    k experts held, the first `train` requests the history every policy
    starts from."""
    history = collections.Counter()
    for counts, _, _ in reqs[:train]:
        if policy == "adaptive":
            for e in history:
                history[e] *= decay
        history.update(counts)
    held = top(history, k)
    out = []
    for counts, _, prefill in reqs[train:]:
        if policy == "oracle":
            new = top(counts, k)
        elif policy == "prefill":
            new = top(_mix(history, prefill, prefill_weight), k)
        elif policy == "adaptive":
            new = top(history, k)
        else:
            new = held
        moved = len(new - held)
        held = new
        total = sum(counts.values())
        out.append((sum(n for e, n in counts.items() if e in held) / total, moved))
        if policy in ("adaptive", "prefill"):
            for e in history:
                history[e] *= decay
            history.update(counts)
    return out


def _mix(history, prefill, weight):
    h, p = sum(history.values()) or 1, sum(prefill.values()) or 1
    return {e: (1 - weight) * history.get(e, 0) / h + weight * prefill.get(e, 0) / p for e in history.keys() | prefill.keys()}


def cpu_ms(share, pairs_per_step):
    return CPU_FIXED_MS + CPU_PAIR_MS * (1 - share) * pairs_per_step


def run(reqs, per_expert_bytes, budgets_gb, train_fraction=0.3, decays=DECAYS, pcie_gb_per_s=25.0,
        prefill_weights=PREFILL_WEIGHTS):
    """{budget: [row]}: one row per policy (and decay), as report prints."""
    train = max(1, int(len(reqs) * train_fraction))
    if train >= len(reqs):
        raise SystemExit(f"{len(reqs)} requests leave none to evaluate after a training prefix of {train}")
    evaluated = reqs[train:]
    selections = sum(sum(c.values()) for c, _, _ in evaluated)
    steps = sum(s for _, s, _ in evaluated)
    pairs_per_step = selections / steps if steps else 0.0
    out = {}
    for gb in budgets_gb:
        k = int(gb * 1e9 // per_expert_bytes)
        rows = []
        policies = ([("static", "static", 1.0, 0)] + [(f"adaptive {d:g}", "adaptive", d, 0) for d in decays]
                    + [(f"prefill {w:g}", "prefill", 1.0, w) for w in prefill_weights] + [("oracle", "oracle", 1.0, 0)])
        for label, policy, decay, weight in policies:
            got = simulate(reqs, k, train, policy, decay, weight)
            shares = [s for s, _ in got]
            overall = sum(s * sum(c.values()) for s, (c, _, _) in zip(shares, evaluated)) / selections
            moved = statistics.mean(m for _, m in got)
            moved_gb = moved * per_expert_bytes / 1e9
            per_req_steps = steps / len(evaluated) if steps else 0
            rows.append({"policy": label, "experts": k, "share": round(overall, 4), "share_min": round(min(shares), 4),
                         "share_median": round(statistics.median(shares), 4), "moved_per_request": round(moved, 1),
                         "moved_gb_per_request": round(moved_gb, 2),
                         "copy_ms_per_request": round(moved_gb / pcie_gb_per_s * 1e3, 1),
                         "copy_ms_per_token": round(moved_gb / pcie_gb_per_s * 1e3 / per_req_steps, 2) if per_req_steps else None,
                         "cpu_ms_per_token": round(cpu_ms(overall, pairs_per_step), 2) if steps else None})
        out[gb] = rows
    return out, {"requests": len(reqs), "train": train, "evaluated": len(evaluated),
                 "pairs_per_step": round(pairs_per_step, 1), "pcie_gb_per_s": pcie_gb_per_s}


def markdown_table(headers, rows):
    """Format an aligned Markdown table."""
    header = "| " + " | ".join(headers) + " |"
    sep = "|:---|" + "|".join("---:" for _ in headers[1:]) + "|"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def report(results, meta, per_expert_bytes):
    lines = [f"{meta['requests']} requests: placements start from the first {meta['train']}, scored on the other "
             f"{meta['evaluated']}; {meta['pairs_per_step']:g} pairs per decode step; expert {per_expert_bytes / 1e6:.1f} MB; "
             f"copies at an assumed {meta['pcie_gb_per_s']:g} GB/s; CPU ms = {CPU_FIXED_MS} + {CPU_PAIR_MS} x cold pairs"]
    headers = ["policy", "GPU share (all / min / median)", "moved per request", "copy ms (request / token)", "CPU ms/token"]
    for gb, rows in results.items():
        lines.append(f"\n{gb:g} GB ({rows[0]['experts']} experts held):")
        table_rows = [
            [r['policy'], f"{r['share']:.0%} / {r['share_min']:.0%} / {r['share_median']:.0%}",
             f"{r['moved_per_request']:g} ({r['moved_gb_per_request']:g} GB)",
             f"{r['copy_ms_per_request']:g} / {r['copy_ms_per_token']}", str(r['cpu_ms_per_token'])]
            for r in rows
        ]
        lines.append(markdown_table(headers, table_rows))
    return "\n".join(lines)


# --- Per-step trace replay (#348, #447) ---


class Cache:
    """A cache holding up to k items.

    Policies:
      static: holds a fixed set chosen during warm-up; never inserts or evicts.
      lru: evicts the item selected least recently.
      decayed_lfu: evicts the item with the lowest score, decayed each step.
                   With decay 1.0, this is cumulative LFU.
    """

    def __init__(self, k, policy="lru", decay=1.0, static_set=None):
        self.k = k
        self.policy = policy
        self.decay = decay
        self.items = set(static_set) if static_set is not None else set()
        self.scores = collections.defaultdict(float)

    def step(self):
        """Called at each decode step to decay scores."""
        if self.policy == "decayed_lfu" and self.decay < 1.0:
            for item in self.scores:
                self.scores[item] *= self.decay

    def access(self, item, step_idx):
        """Record an access. Returns True if hit, False if miss."""
        hit = item in self.items
        if self.policy == "static":
            return hit

        if self.policy == "lru":
            self.scores[item] = step_idx
        elif self.policy == "decayed_lfu":
            self.scores[item] += 1.0

        if not hit and self.k > 0:
            if len(self.items) >= self.k:
                victim = min(self.items, key=lambda x: (self.scores[x], x))
                self.items.remove(victim)
            self.items.add(item)
        return hit


def traces(docs_or_files, files=None):
    """Extract ordered steps per layer from files or documents carrying expert_trace.

    Records are sorted by time (matching requests()). Refuses records without a trace.
    Records with an empty trace (e.g. prompt ends on EOS) count as zero steps.

    Returns:
        (layer_steps, total_tokens) where layer_steps maps layer index (int)
        to a list of decode steps, each step being a list of selected expert
        indices (ints).
    """
    if docs_or_files and isinstance(docs_or_files[0], str):
        file_doc_pairs = []
        for f in docs_or_files:
            for doc in expert_stats.load([f]):
                file_doc_pairs.append((f, doc))
    elif docs_or_files and isinstance(docs_or_files[0], tuple):
        file_doc_pairs = docs_or_files
    else:
        file_doc_pairs = [(files[i] if files and i < len(files) else "input", doc) for i, doc in enumerate(docs_or_files)]

    records_with_order = []
    found_any_trace = False
    for i, (fname, doc) in enumerate(file_doc_pairs):
        records = expert_stats._records(doc)
        if not records:
            continue
        for tel in records:
            if "expert_trace" not in tel:
                raise SystemExit(f"record in {fname} has no expert trace (record with telemetry=vxpu_pb2.TELEMETRY_TRACE)")
            found_any_trace = True
            trace = tel["expert_trace"]
            time_key = doc.get("time") or ""
            steps_count = tel.get("decode_steps") or (len(next(iter(trace.values()))) if trace else 0)
            if trace:
                records_with_order.append((time_key, i, trace, steps_count))

    if not found_any_trace:
        raise SystemExit("no expert traces found (record with telemetry=vxpu_pb2.TELEMETRY_TRACE)")

    records_with_order.sort(key=lambda r: (r[0], r[1]))

    layer_steps = collections.defaultdict(list)
    total_tokens = 0
    for _, _, trace, steps_count in records_with_order:
        total_tokens += steps_count
        for layer_str, steps in sorted(trace.items(), key=lambda kv: int(kv[0])):
            layer_steps[int(layer_str)].extend(steps)

    lengths = {layer: len(steps) for layer, steps in sorted(layer_steps.items())}
    if len(set(lengths.values())) > 1:
        details = ", ".join(f"layer {l}: {n} steps" for l, n in lengths.items())
        raise SystemExit(f"expert traces have unequal step counts across layers ({details})")

    return dict(layer_steps), total_tokens


def simulate_trace_policy(layer_steps, k, train_steps, policy, decay=1.0):
    """Simulate a cache policy for capacity k experts across all layers on a per-step trace.

    Every policy is scored on the exact same steps (steps[train_steps:]).

    Returns:
        (hits, misses, total_selections)
    """
    hits = 0
    misses = 0

    for layer, steps in sorted(layer_steps.items()):
        total_steps = len(steps)
        if policy == "oracle":
            eval_counts = collections.Counter(e for step in steps[train_steps:] for e in step)
            ranked = sorted(eval_counts.values(), reverse=True)
            layer_hits = sum(ranked[:k])
            layer_sel = sum(eval_counts.values())
            hits += layer_hits
            misses += (layer_sel - layer_hits)
            continue

        if policy == "static":
            warm_counts = collections.Counter(e for step in steps[:train_steps] for e in step)
            static_set = {e for e, _ in warm_counts.most_common(k)}
            cache = Cache(k, policy="static", static_set=static_set)
        else:
            cache = Cache(
                k,
                policy="lru" if policy == "lru" else "decayed_lfu",
                decay=decay if policy == "decayed_lfu" else 1.0,
            )

        for s in range(total_steps):
            cache.step()
            for e in steps[s]:
                hit = cache.access(e, s)
                if s >= train_steps:
                    if hit:
                        hits += 1
                    else:
                        misses += 1

    return hits, misses, hits + misses


def simulate_two_tier(layer_steps, k_gpu, gpu_policy, k_ram, ram_policy, train_steps):
    """Simulate two cache tiers: GPU (fastest) and RAM (host cache over storage).

    Tiers are inclusive: RAM holds the model banks, so every selection reaches both
    tiers. An access is served by the GPU if it hits in the GPU, by RAM if it hits in
    RAM, and by storage otherwise.

    Returns:
        dict with counts and shares served by 'gpu', 'ram', and 'storage'.
    """
    served = collections.Counter()
    total_post = 0

    for layer, steps in sorted(layer_steps.items()):
        warm_counts = collections.Counter(e for step in steps[:train_steps] for e in step)
        gpu_static = {e for e, _ in warm_counts.most_common(k_gpu)} if gpu_policy == "static" else None
        ram_static = {e for e, _ in warm_counts.most_common(k_ram)} if ram_policy == "static" else None

        gpu_cache = Cache(
            k_gpu,
            policy="static" if gpu_policy == "static" else ("lru" if gpu_policy == "lru" else "decayed_lfu"),
            static_set=gpu_static,
        )
        ram_cache = Cache(
            k_ram,
            policy="static" if ram_policy == "static" else ("lru" if ram_policy == "lru" else "decayed_lfu"),
            static_set=ram_static,
        )

        for s, step in enumerate(steps):
            is_after = s >= train_steps
            gpu_cache.step()
            ram_cache.step()

            for e in step:
                if is_after:
                    total_post += 1

                in_gpu = gpu_cache.access(e, s)
                in_ram = ram_cache.access(e, s)

                if in_gpu:
                    if is_after:
                        served["gpu"] += 1
                elif in_ram:
                    if is_after:
                        served["ram"] += 1
                else:
                    if is_after:
                        served["storage"] += 1

    gpu_share = served["gpu"] / total_post if total_post else 0.0
    ram_share = served["ram"] / total_post if total_post else 0.0
    storage_share = served["storage"] / total_post if total_post else 0.0

    return {
        "total_evaluated": total_post,
        "served": dict(served),
        "gpu_share": gpu_share,
        "ram_share": ram_share,
        "storage_share": storage_share,
    }


def simulate_prefetch(layer_steps, train_steps, per_expert_bytes=None, tokens_per_s=20.0):
    """Evaluate prefetch predictor: same experts as the last decode step."""
    tp, fp, fn = 0, 0, 0
    total_steps = len(next(iter(layer_steps.values()))) if layer_steps else 0
    start = max(1, train_steps)
    for s in range(start, total_steps):
        for l, steps in layer_steps.items():
            actual = set(steps[s])
            predicted = set(steps[s - 1])
            tp += len(predicted & actual)
            fp += len(predicted - actual)
            fn += len(actual - predicted)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    eval_steps = max(0, total_steps - start)
    bw_saved = (tp / eval_steps) * per_expert_bytes * tokens_per_s / 1e9 if per_expert_bytes and eval_steps else None
    return {"precision": prec, "recall": rec, "bandwidth_saved_gb_per_s": bw_saved}


def gb_to_experts_per_layer(gb, num_layers, per_expert_bytes):
    """Convert a GB budget across all layers into experts per layer.

    Refuses budgets that fit less than one expert per layer (except 0).
    """
    if gb == 0:
        return 0
    k = int(gb * 1e9 // (num_layers * per_expert_bytes))
    if k < 1:
        min_gb = (num_layers * per_expert_bytes) / 1e9
        raise SystemExit(f"budget {gb:g} GB is smaller than one expert per layer ({min_gb:.2f} GB needed)")
    return k


def resolve_expert_size(artifact, expert_mb):
    """Resolve expert size in bytes and expert count from artifact or explicit MB."""
    if expert_mb:
        return expert_mb * 1e6, None
    if artifact:
        sized = expert_stats.expert_bytes(artifact)
        if sized:
            return sized
    return None, None


def report_trace(docs_or_files, per_expert, experts=None, budgets_gb="60,165",
                 train=0.3, tokens_per_s=20.0, gpu_gb=60.0, gpu_policy="lru",
                 ram_gb=165.0, ram_policy="lru", decays=DECAYS):
    """Run trace cache simulation over telemetry documents and format the report.

    Returns:
        (out_dict, report_text)
    """
    layer_steps, total_tokens = traces(docs_or_files)
    num_layers = len(layer_steps)
    if not experts:
        experts = max(max(e for step in steps for e in step) for steps in layer_steps.values()) + 1

    total_selections = sum(sum(len(step) for step in steps) for steps in layer_steps.values())
    total_steps = len(next(iter(layer_steps.values())))

    if not (0.0 <= train < 1.0):
        raise SystemExit(f"--train must be a fraction in [0, 1), got {train}")
    train_steps = int(round(total_steps * train))
    evaluated_steps = total_steps - train_steps
    if evaluated_steps <= 0:
        raise SystemExit(f"{total_steps} steps leave none to evaluate after a training prefix of {train_steps}")

    gbs = [float(b.strip()) for b in budgets_gb.split(",")] if isinstance(budgets_gb, str) else [float(b) for b in budgets_gb]
    # Build capacity list: [(k, label, key)]
    capacities_list = []
    for gb in list(dict.fromkeys(gbs)):
        k = min(experts, gb_to_experts_per_layer(gb, num_layers, per_expert))
        capacities_list.append((k, f"{gb:g} GB ({k} experts held)", f"{gb:g}"))

    policies = (
        [("static", "static", 1.0), ("lru", "lru", 1.0)]
        + [(f"decayed lfu {d:g}", "decayed_lfu", d) for d in decays]
        + [("oracle", "oracle", 1.0)]
    )

    # 1. Single-Tier Policies (all scored on steps[train_steps:])
    policy_results = {}
    for label, pol, d in policies:
        rates, hits_dict, misses_dict, bws = {}, {}, {}, {}
        for k, _, key in capacities_list:
            h, m, total = simulate_trace_policy(layer_steps, k, train_steps, pol, decay=d)
            rate = h / total if total else 0.0
            rates[key], hits_dict[key], misses_dict[key] = round(rate, 4), h, m
            if total_tokens:
                miss_bytes = (1.0 - rate) * (total_selections / total_tokens) * per_expert
                bws[key] = round(miss_bytes * tokens_per_s / 1e9, 2)

        policy_results[label] = {"hit_rate": rates, "hits": hits_dict, "misses": misses_dict, "bandwidth_gb_per_s": bws}

    # 2. Two-Tier Simulation
    k_gpu = min(experts, gb_to_experts_per_layer(gpu_gb, num_layers, per_expert))
    k_ram = min(experts, gb_to_experts_per_layer(ram_gb, num_layers, per_expert))
    two_tier_res = simulate_two_tier(layer_steps, k_gpu, gpu_policy, k_ram, ram_policy, train_steps)
    if total_tokens:
        two_tier_miss_bytes = two_tier_res["storage_share"] * (total_selections / total_tokens) * per_expert
        two_tier_res["storage_bandwidth_gb_per_s"] = round(two_tier_miss_bytes * tokens_per_s / 1e9, 2)

    # 3. Prefetch Predictor Simulation
    prefetch_res = simulate_prefetch(layer_steps, train_steps, per_expert_bytes=per_expert, tokens_per_s=tokens_per_s)

    out = {
        "layers": num_layers,
        "experts_per_layer": experts,
        "selections": total_selections,
        "decode_steps": total_steps,
        "train_steps": train_steps,
        "evaluated_steps": evaluated_steps,
        "budgets_gb": [key for _, _, key in capacities_list],
        "policies": policy_results,
        "two_tier": {
            "gpu_gb": gpu_gb,
            "gpu_policy": gpu_policy,
            "ram_gb": ram_gb,
            "ram_policy": ram_policy,
            "gpu_share": round(two_tier_res["gpu_share"], 4),
            "ram_share": round(two_tier_res["ram_share"], 4),
            "storage_share": round(two_tier_res["storage_share"], 4),
            "storage_bandwidth_gb_per_s": two_tier_res["storage_bandwidth_gb_per_s"],
        },
        "prefetch": {
            "precision": round(prefetch_res["precision"], 4),
            "recall": round(prefetch_res["recall"], 4),
            **({"bandwidth_saved_gb_per_s": round(prefetch_res["bandwidth_saved_gb_per_s"], 2)} if prefetch_res["bandwidth_saved_gb_per_s"] is not None else {})
        },
    }

    # Format text output
    lines = [
        f"decode: {total_selections} expert selections over {num_layers} layers, "
        f"{experts} experts per layer; {total_steps} decode steps "
        f"(warm-up: {train_steps} steps, evaluated: {evaluated_steps} steps)",
    ]
    if total_selections / max(1, num_layers * experts) < 20:
        lines.append(
            f"  note: {int(total_selections / max(1, num_layers))} selections per layer "
            f"against {experts} experts is a small sample; unseen experts look cold, so skew is overstated"
        )

    lines.append("\nexpert cache policies: share of selections served per budget:")
    col_headers = [label for _, label, _ in capacities_list]
    table_rows = [
        [label] + [f"{int(round(policy_results[label]['hit_rate'][key] * 100)):>4}%" for _, _, key in capacities_list]
        for label, _, _ in policies
    ]
    lines.append(markdown_table(["policy"] + [f"{h:>5}" for h in col_headers], table_rows))

    size_str = f"{per_expert / 1e6:.1f} MB" if per_expert >= 1e5 else f"{per_expert / 1e3:.1f} KB"
    lines.append(
        f"\nstorage bandwidth for the misses at {tokens_per_s:g} tokens/s "
        f"(expert size {size_str}, {total_selections / max(1, total_tokens):.1f} selections/token):"
    )
    bw_headers = [f"{gb:g} GB" for gb in list(dict.fromkeys(gbs))]
    bw_rows = [
        [label] + [f"{policy_results[label]['bandwidth_gb_per_s'][key]:>5.2f} GB/s" for _, _, key in capacities_list]
        for label, _, _ in policies
    ]
    lines.append(markdown_table(["policy"] + [f"{h:>5}" for h in bw_headers], bw_rows))

    lines.append(f"\ntwo tiers: GPU ({gpu_gb:g} GB capacity, {gpu_policy}), RAM ({ram_gb:g} GB capacity, {ram_policy}):\n"
                 f"  share served: GPU {two_tier_res['gpu_share'] * 100:.1f}%, "
                 f"RAM {two_tier_res['ram_share'] * 100:.1f}%, "
                 f"Storage {two_tier_res['storage_share'] * 100:.1f}%\n"
                 f"  storage bandwidth at {tokens_per_s:g} tokens/s: {two_tier_res['storage_bandwidth_gb_per_s']:.1f} GB/s")

    bw_str = f", bandwidth saved: {prefetch_res['bandwidth_saved_gb_per_s']:.1f} GB/s" if prefetch_res["bandwidth_saved_gb_per_s"] is not None else ""
    lines.append(f"\nprefetch predictor (same experts as last step): precision {prefetch_res['precision'] * 100:.1f}%, recall {prefetch_res['recall'] * 100:.1f}%{bw_str}")

    return out, "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    # Subcommand: requests
    p_req = sub.add_parser("requests", help="replay request-level summary records against placement policies")
    p_req.add_argument("files", nargs="+", help="request records (`vxpu stats --records`) or other telemetry files")
    p_req.add_argument("--artifact", help="artifact directory, for the expert size")
    p_req.add_argument("--expert-mb", type=float, help="the expert size without an artifact (DeepSeek-V4.1-Flash: 18.8)")
    p_req.add_argument("--budgets-gb", default="60,165", help="comma-separated GPU budgets in GB (default: 60,165)")
    p_req.add_argument("--train", type=float, default=0.3, help="the fraction of requests every placement starts from")
    p_req.add_argument("--decays", default=",".join(f"{d:g}" for d in DECAYS))
    p_req.add_argument("--prefill-weights", default=",".join(f"{w:g}" for w in PREFILL_WEIGHTS))
    p_req.add_argument("--pcie-gb-per-s", type=float, default=25.0, help="assumed host-to-device copy bandwidth")
    p_req.add_argument("--json", help="also write the results here")

    # Subcommand: steps
    p_step = sub.add_parser("steps", help="replay per-step ordered selection traces through cache policies")
    p_step.add_argument("files", nargs="+", help="telemetry JSON files with expert traces")
    p_step.add_argument("--artifact", help="artifact directory, for the expert size")
    p_step.add_argument("--expert-mb", type=float, help="the expert size without an artifact (DeepSeek-V4.1-Flash: 18.8)")
    p_step.add_argument("--budgets-gb", default="60,165", help="comma-separated GPU budgets in GB (default: 60,165)")
    p_step.add_argument("--tokens-per-s", type=float, default=20.0, help="target tokens/s for storage bandwidth")
    p_step.add_argument("--train", type=float, default=0.3, help="the fraction of steps every placement starts from (in [0, 1))")
    p_step.add_argument("--gpu-gb", type=float, default=60.0, help="GPU tier budget in GB for two-tier simulation (default: 60)")
    p_step.add_argument("--gpu-policy", default="lru", choices=["lru", "lfu", "static"], help="policy for GPU tier")
    p_step.add_argument("--ram-gb", type=float, default=165.0, help="RAM tier budget in GB for two-tier simulation (default: 165)")
    p_step.add_argument("--ram-policy", default="lru", choices=["lru", "lfu", "static"], help="policy for RAM tier")
    p_step.add_argument("--json", help="also write the results here")

    args = p.parse_args(argv)
    per_expert, experts = resolve_expert_size(args.artifact, args.expert_mb)
    if not per_expert:
        raise SystemExit("--artifact or --expert-mb is needed for the expert size")

    if args.command == "steps":
        file_doc_pairs = []
        for f in args.files:
            for doc in expert_stats.load([f]):
                file_doc_pairs.append((f, doc))

        out, text = report_trace(
            file_doc_pairs,
            per_expert,
            experts=experts,
            budgets_gb=args.budgets_gb,
            train=args.train,
            tokens_per_s=args.tokens_per_s,
            gpu_gb=args.gpu_gb,
            gpu_policy=args.gpu_policy,
            ram_gb=args.ram_gb,
            ram_policy=args.ram_policy,
        )
        print(text)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=1)
    else:
        docs = expert_stats.load(args.files)
        budgets = [float(b) for b in args.budgets_gb.split(",")]
        reqs = requests(docs)
        results, meta = run(
            reqs, per_expert, budgets, args.train,
            [float(d) for d in args.decays.split(",")], args.pcie_gb_per_s,
            [float(w) for w in args.prefill_weights.split(",")]
        )
        print(report(results, meta, per_expert))
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"meta": meta, "budgets": {str(k): v for k, v in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
