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

"""Expert usage statistics from execution telemetry (#317).

Which experts a mixture-of-experts model actually selects decides where
they should live. With even usage, a machine with less RAM than the banks
pages experts from storage at a predictable rate; with skew, a hot set fits
in RAM (or on the GPU) and the rest can stream. This reads the expert
counts the engine's telemetry records (telemetry.py: per phase, per layer,
how often each expert was chosen) and reports, per layer and overall:

  selections   how many expert selections were seen
  distinct     how many different experts were selected
  entropy      bits, against log2(experts) for perfectly even use
  hit rate     the share of selections a cache holding the hottest
               fraction f of each layer's experts would have served (an
               oracle cache chosen in hindsight: an upper bound for any
               real policy on this sample)
  even use     the same oracle's hit rate on a sample of the same size
               drawn with perfectly even use: what the hit rate would be
               from sampling noise alone. Skew is the gap between the two.

and, with --artifact, the expert size and the storage bandwidth the misses
would need at a target decode rate.

Counts come from any file carrying telemetry: `vectors record --telemetry`
output (each prompt's "telemetry"), attribute_decode.py output, or a raw
telemetry dict. Small samples overstate skew: an expert that was not seen
looks cold. The report says when the sample is small against the number
of experts.

    python -m vxpu.expert_stats FILE... [--phase decode] [--artifact DIR] [--tokens-per-s 20]
"""

import argparse
import collections
import json
import math
import os
import random

FRACTIONS = (0.25, 0.5, 0.75, 0.85, 0.9, 0.95)


def collect(docs):
    """{phase: {layer: Counter(expert -> count)}} summed over every
    telemetry record found in docs (parsed JSON documents)."""
    out = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    def add(tel):
        for phase, layers in (tel.get("experts") or {}).items():
            for layer, counts in layers.items():
                for expert, n in counts.items():
                    out[phase][int(layer)][int(expert)] += n

    for doc in docs:
        if doc.get("format", "").startswith("vxpu-telemetry/"):
            add(doc)
        if isinstance(doc.get("telemetry"), dict):  # attribute_decode.py output
            add(doc["telemetry"])
        for prompt in doc.get("prompts", []):  # vectors files
            if isinstance(prompt.get("telemetry"), dict):
                add(prompt["telemetry"])
    return out


def expert_bytes(artifact_dir):
    """Bytes of one expert (its slice of every bank: weights and scales),
    and the number of experts per layer, from an artifact's binding and
    manifest. None when the artifact places no expert banks externally."""
    with open(os.path.join(artifact_dir, "binding.json")) as f:
        binding = json.load(f)
    with open(os.path.join(artifact_dir, "manifest.json")) as f:
        tensors = json.load(f)["tensors"]
    layers = (binding.get("external") or {}).get("experts") or {}
    if not layers:
        return None
    specs = next(iter(layers.values()))
    experts = len(next(iter(specs.values()))["parts"])
    total = sum(tensors[name]["length"] for spec in specs.values() for part in spec["parts"] for name in part)
    return total / experts, experts


def entropy(counts):
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else 0.0


def oracle_hits(counts, experts):
    """{fraction: share of the selections the hottest fraction serves}."""
    n = sum(counts.values())
    ranked = sorted(counts.values(), reverse=True)
    return {f: sum(ranked[:math.ceil(f * experts)]) / n for f in FRACTIONS}


def even_sample(n, experts, rng):
    """Counts of n selections drawn with every expert equally likely."""
    return collections.Counter(rng.randrange(experts) for _ in range(n))


def analyse(layers, experts, seed=0):
    """Per-layer and overall statistics for one phase's counts, with the
    even-use baseline at the same sample size."""
    rng = random.Random(seed)
    rows = {}
    overall_hits, overall_even = collections.Counter(), collections.Counter()
    total = 0
    for layer, counts in sorted(layers.items()):
        n = sum(counts.values())
        e = experts
        hits = oracle_hits(counts, e)
        even = oracle_hits(even_sample(n, e, rng), e)
        for f in FRACTIONS:
            overall_hits[f] += hits[f] * n
            overall_even[f] += even[f] * n
        total += n
        rows[layer] = {"selections": n, "distinct": len(counts), "entropy_bits": round(entropy(counts), 3),
                       "even_bits": round(math.log2(e), 3), "hit_rate": {str(f): round(h, 4) for f, h in hits.items()},
                       "hit_rate_even_use": {str(f): round(h, 4) for f, h in even.items()}}
    overall = {str(f): round(overall_hits[f] / total, 4) for f in FRACTIONS} if total else {}
    overall_even = {str(f): round(overall_even[f] / total, 4) for f in FRACTIONS} if total else {}
    return rows, overall, overall_even, total


def report(paths, phase="decode", artifact=None, tokens_per_s=20.0):
    docs = []
    for path in paths:
        with open(path) as f:
            docs.append(json.load(f))
    counts = collect(docs)
    if phase not in counts:
        raise SystemExit(f"no {phase} expert counts in {', '.join(paths)} (record with telemetry)")
    sized = expert_bytes(artifact) if artifact else None
    # without the artifact, the experts per layer are at least the highest
    # index any phase selected, plus one
    experts = sized[1] if sized else 1 + max(e for layers in counts.values() for c in layers.values() for e in c)
    rows, overall, overall_even, total = analyse(counts[phase], experts)
    layers = len(rows)
    out = {"phase": phase, "layers": layers, "selections": total, "experts_per_layer": experts,
           "layers_detail": rows, "hit_rate_overall": overall, "hit_rate_even_use": overall_even}
    lines = [f"{phase}: {total} expert selections over {layers} layers"
             + f", {experts} experts per layer" + ("" if sized else " (the highest index seen; --artifact for the binding's)")]
    per_layer = total / max(layers, 1)
    if per_layer < 20 * experts:
        lines.append(f"  note: {per_layer:.0f} selections per layer against {experts} experts is a small sample; "
                     "unseen experts look cold, so skew is overstated")
    ent = sum(r["entropy_bits"] for r in rows.values()) / max(layers, 1)
    even = sum(r["even_bits"] for r in rows.values()) / max(layers, 1)
    lines.append(f"  entropy: {ent:.2f} bits on average against {even:.2f} for even use")
    lines.append("  oracle cache of the hottest fraction of each layer's experts -> share of selections served:")
    lines.append("    observed: " + "  ".join(f"{float(f):.0%}: {h:.0%}" for f, h in overall.items()))
    lines.append("    even use: " + "  ".join(f"{float(f):.0%}: {h:.0%}" for f, h in overall_even.items())
                 + "   (the same sample size with every expert equally likely)")
    if sized:
        per_expert = sized[0]
        tokens = _tokens(docs, phase)
        if not tokens:
            raise SystemExit("cannot size the misses: the telemetry carries no token count")
        selections_per_token = total / tokens
        out["expert_bytes"] = per_expert
        bw = {}
        for f, h in overall.items():
            miss_bytes = (1 - h) * selections_per_token * per_expert
            bw[f] = miss_bytes * tokens_per_s / 1e9
        out["storage_gb_per_s_at_target"] = {"tokens_per_s": tokens_per_s, **{f: round(v, 2) for f, v in bw.items()}}
        lines.append(f"  expert size {per_expert / 1e6:.1f} MB; {selections_per_token:.0f} selections per token; "
                     f"storage bandwidth for the misses at {tokens_per_s:g} tokens/s:")
        lines.append("    " + "  ".join(f"{float(f):.0%} cached: {v:.1f} GB/s" for f, v in bw.items()))
    return out, "\n".join(lines)


def _tokens(docs, phase):
    """Decode steps (or prompt tokens for prefill) the counts cover."""
    n = 0
    for doc in docs:
        tels = []
        if doc.get("format", "").startswith("vxpu-telemetry/"):
            tels.append(doc)
        if isinstance(doc.get("telemetry"), dict):
            tels.append(doc["telemetry"])
        tels += [p["telemetry"] for p in doc.get("prompts", []) if isinstance(p.get("telemetry"), dict)]
        for tel in tels:
            n += tel.get("decode_steps", 0) if phase == "decode" else 0
        if phase == "prefill":
            n += sum(len(p.get("input_ids", [])) for p in doc.get("prompts", []) if isinstance(p.get("telemetry"), dict))
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("files", nargs="+", help="vectors files (--telemetry), attribute_decode.py output, or telemetry JSON")
    p.add_argument("--phase", default="decode", choices=("decode", "prefill"))
    p.add_argument("--artifact", help="artifact directory, for expert sizes and storage bandwidth")
    p.add_argument("--tokens-per-s", type=float, default=20.0)
    p.add_argument("--json", help="also write the full report here")
    args = p.parse_args(argv)
    out, text = report(args.files, args.phase, args.artifact, args.tokens_per_s)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
