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

  static     the hottest experts of a training prefix of the requests,
             held for the rest (what VXPU_GPU_EXPERTS does today)
  adaptive   decayed LFU: every expert's score is multiplied by the decay
             after each request and its selections added; before each
             request the top scores are held. Decay 1 is cumulative LFU
             (the static placement, kept up to date); 0 holds the last
             request's experts. Starts from the training prefix too.
  prefill    the request's own prefill (its prompt's selections, known
             before the first decode step) mixed with the cumulative
             history, each normalized to sum 1, the prefill weighed by
             one of --prefill-weights: what the prompt says about its own
             decode. Its moves happen after the prefill, before decode
  oracle     each request's own hottest decode experts, chosen in
             hindsight: an upper bound for any policy that places once
             per request

Every policy is scored on the requests after the training prefix, at
each budget (GB of GPU memory; every expert costs the same bytes, so a
budget is a count of experts across all layers):

  GPU share  the share of the request's selections the held experts serve
  moved      experts copied in before the request (a policy that moves
             frees as many), in experts and GB, and the copy time at an
             assumed host-to-device bandwidth (--pcie-gb-per-s)
  CPU ms     an estimate of the CPU's expert time per decode step: a
             linear model, fixed + per pair x cold pairs, fitted to the
             measurements of #368 on vxpu-hero (21.6 ms with all 240
             pairs on the CPU; 8.6 to 11.1 ms with 59-71% on the GPU).
             It does not model the GPU's side, which runs concurrently
             and is shorter at these shares

The records carry counts per request, not the order of the selections
within one, so this simulates moving experts between requests only;
moving them within a request (as a conversation changes topic) needs a
per-step trace.

    python -m vxpu.expert_sim RECORDS... [--artifact DIR | --expert-mb 18.8] [--budgets-gb 60,165]
"""

import argparse
import collections
import json
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


def report(results, meta, per_expert_bytes):
    lines = [f"{meta['requests']} requests: placements start from the first {meta['train']}, scored on the other "
             f"{meta['evaluated']}; {meta['pairs_per_step']:g} pairs per decode step; expert {per_expert_bytes / 1e6:.1f} MB; "
             f"copies at an assumed {meta['pcie_gb_per_s']:g} GB/s; CPU ms = {CPU_FIXED_MS} + {CPU_PAIR_MS} x cold pairs"]
    for gb, rows in results.items():
        lines.append(f"\n{gb:g} GB ({rows[0]['experts']} experts held):")
        lines.append("| policy | GPU share (all / min / median) | moved per request | copy ms (request / token) | CPU ms/token |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r['policy']} | {r['share']:.0%} / {r['share_min']:.0%} / {r['share_median']:.0%} "
                         f"| {r['moved_per_request']:g} ({r['moved_gb_per_request']:g} GB) "
                         f"| {r['copy_ms_per_request']:g} / {r['copy_ms_per_token']} | {r['cpu_ms_per_token']} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("files", nargs="+", help="request records (`vxpu stats --records`) or other telemetry files")
    p.add_argument("--artifact", help="artifact directory, for the expert size")
    p.add_argument("--expert-mb", type=float, help="the expert size without an artifact (DeepSeek-V4.1-Flash: 18.8)")
    p.add_argument("--budgets-gb", default="60,165", help="comma-separated GPU budgets")
    p.add_argument("--train", type=float, default=0.3, help="the fraction of requests every placement starts from")
    p.add_argument("--decays", default=",".join(f"{d:g}" for d in DECAYS))
    p.add_argument("--prefill-weights", default=",".join(f"{w:g}" for w in PREFILL_WEIGHTS))
    p.add_argument("--pcie-gb-per-s", type=float, default=25.0, help="assumed host-to-device copy bandwidth")
    p.add_argument("--json", help="also write the results here")
    args = p.parse_args(argv)
    if args.artifact:
        per_expert = expert_stats.expert_bytes(args.artifact)[0]
    elif args.expert_mb:
        per_expert = args.expert_mb * 1e6
    else:
        raise SystemExit("--artifact or --expert-mb is needed for the expert size")
    reqs = requests(expert_stats.load(args.files))
    results, meta = run(reqs, per_expert, [float(b) for b in args.budgets_gb.split(",")], args.train,
                        [float(d) for d in args.decays.split(",")], args.pcie_gb_per_s,
                        [float(w) for w in args.prefill_weights.split(",")])
    print(report(results, meta, per_expert))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "budgets": {str(k): v for k, v in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
