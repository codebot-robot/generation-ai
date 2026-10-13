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

"""Test vectors: what a model says, recorded by any producer, compared.

A vectors file is the interchange format between engines. Eager
transformers records the reference; the vXPU engine (in-process, or
through a router) records a candidate; a vLLM backend would record one
too. ``compare`` diffs two files: greedy tokens must agree exactly, and
where both sides carry logits the top-k values must agree within a
tolerance.

    python -m vxpu.vectors record --producer transformers --model REPO -o ref.json
    python -m vxpu.vectors record --producer vxpu-engine --model REPO --artifact DIR -o got.json
    python -m vxpu.vectors record --producer vxpu-router --model REPO --artifact DIR [--router host:port] -o got.json
    python -m vxpu.vectors compare ref.json got.json
    python -m vxpu.vectors selftest  # the CI tier: tiny models, in-process

Format (vxpu-vectors/1):

    {"format": "vxpu-vectors/1",
     "model": {"repo": ..., "revision": ..., "commit_sha": ...},
     "producer": {"kind": "transformers" | "vxpu-engine" | "vxpu-router",
                  "torch": ..., "transformers": ..., "device": ..., ...},
     "settings": {"new_tokens": 32, "top_k": 32, "logit_positions": 4,
                  "greedy": true},
     "prompts": [{"name": ..., "text": ... | "ids": [...],
                  "input_ids": [...], "generated_ids": [...],
                  "finish": "eos" | "length",
                  "top_logits": [{"ids": [...], "values": [...]}, ...],
                  "prefill_ms": ..., "ms_per_token": ...}]}

Logits are stored as the top-k (id, value) pairs of the first few
generated positions: enough to compare two engines' views of the same
distribution, small enough to live next to a test.
"""

import argparse
import json
import os
import tempfile
import time

PROMPTS = [
    {"name": "sky", "text": "Is the sky blue?"},
    {"name": "kubernetes", "text": "Write one sentence about Kubernetes."},
    {"name": "long", "text": (
        "Kubernetes schedules containers onto nodes, restarts them when "
        "they fail, and scales them with demand. ") * 24
        + "Summarize the above in one sentence."},
]

# Prompts as token ids, for models without a tokenizer (test fixtures).
ID_PROMPTS = [
    {"name": "ids-short", "ids": [5, 17, 42, 99, 7, 300, 11, 128, 64, 2, 250, 33, 8]},
    {"name": "ids-long", "ids": [(i * 37 + 11) % 500 + 3 for i in range(100)]},
]

DEFAULTS = {"new_tokens": 32, "top_k": 32, "logit_positions": 4, "greedy": True}
FORMAT = "vxpu-vectors/1"


def _versions(device):
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "device": str(device)}


def _r(value):
    return None if value is None else round(float(value), 1)


def _top(logits, k):
    import torch
    values, ids = torch.topk(logits.detach().float().cpu(), k)
    return {"ids": ids.tolist(), "values": [round(v, 5) for v in values.tolist()]}


def _encode(tokenizer, prompt):
    """input_ids for a prompt: text through the chat template, or raw ids."""
    if "ids" in prompt:
        return list(prompt["ids"])
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            add_generation_prompt=True, return_tensors="pt",
            return_dict=True)["input_ids"][0].tolist()
    return tokenizer(prompt["text"])["input_ids"]  # base model: plain text


def _tokenizer(model, revision, prompts="auto"):
    if prompts == "ids":
        return None
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model, revision=revision)
    except Exception:  # no tokenizer files, or a backend missing: ids prompts
        return None


def _prompt_set(tokenizer, prompts):
    if prompts == "ids" or (prompts == "auto" and tokenizer is None):
        return ID_PROMPTS
    return PROMPTS


def record_transformers(model, revision, prompts, settings, device="cpu", dtype="auto",
                        dequantize=False):
    """Reference: eager transformers, greedy, logits from generate().

    ``dtype="auto"`` runs the model in the dtype its config declares,
    which is what the exporter captures; a different reference dtype
    would compare two different computations. ``dequantize`` asks
    transformers to expand an fp8 checkpoint to bf16 at load (the same
    weight values, materialized once), which is far faster on a CPU
    than dequantizing per step.
    """
    import torch
    from transformers import AutoModelForCausalLM
    tok = _tokenizer(model, revision, prompts)
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    kwargs = {"dtype": torch_dtype, "revision": revision}
    if dequantize:
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    if device != "cpu":
        kwargs["device_map"] = device
    lm = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()
    eos = lm.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        inputs = torch.tensor([ids], device=lm.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = lm.generate(inputs, max_new_tokens=settings["new_tokens"],
                                 do_sample=False, output_logits=True,
                                 return_dict_in_generate=True)
        elapsed = time.perf_counter() - t0
        new = result.sequences[0, len(ids):].tolist()
        top = [_top(step[0], settings["top_k"])
               for step in result.logits[:settings["logit_positions"]]]
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": "eos" if new and new[-1] in eos_ids else "length",
                    "top_logits": top,
                    "ms_per_token": round(1000 * elapsed / max(1, len(new)), 1)})
    producer = {"kind": "transformers", **_versions(lm.device), "dtype": str(lm.dtype),
                "dequantize": dequantize,
                "experts_implementation": getattr(lm.config, "_experts_implementation", None)}
    return _document(model, revision, lm.config, producer, settings, out)


def record_vxpu_engine(model, revision, prompts, settings, artifact, device="cpu",
                       compile_decode=False):
    """Candidate: the executor's engine in this process (no cluster)."""
    import torch
    from .engine import Engine, SamplingParams
    tok = _tokenizer(model, revision, prompts)
    engine = Engine(artifact, device=device, compile_decode=compile_decode)
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        sid = engine.new_session()
        top = []

        def sink(position, logits, top=top):
            if position < settings["logit_positions"]:
                top.append(_top(logits, settings["top_k"]))

        new, stats = [], None
        for event in engine.generate(sid, ids, settings["new_tokens"],
                                     SamplingParams(do_sample=False),
                                     logits_sink=sink):
            new.extend(event.get("token_ids", []))
            if event.get("done"):
                stats = event
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": stats.get("finish_reason") if stats else None,
                    "top_logits": top,
                    "prefill_ms": stats.get("prefill_ms") if stats else None,
                    "ms_per_token": _r(stats.get("ms_per_token")) if stats else None})
    with open(os.path.join(artifact, "binding.json")) as f:
        binding = json.load(f)
    producer = {"kind": "vxpu-engine", **_versions(device),
                "compile": compile_decode,
                "binding": {k: len(binding.get(k, [])) for k in
                            ("bound", "derived", "state", "stacked", "aliases")},
                "quantization": binding.get("quantization")}
    return _document(model, revision, engine.config, producer, settings, out)


def record_vxpu_router(model, revision, prompts, settings, artifact, router=None):
    """Candidate: an executor reached through a router (tokens only;
    the wire carries no logits)."""
    from .modeling import VxpuModelForCausalLM
    tok = _tokenizer(model, revision, prompts)
    lm = VxpuModelForCausalLM.from_pretrained(
        model, router=router, artifact_dir=artifact, revision=revision,
        progress=lambda *a: None)
    out = []
    try:
        for prompt in _prompt_set(tok, prompts):
            ids = _encode(tok, prompt)
            lm.new_session()
            import torch
            result = lm.generate(torch.tensor([ids]),
                                 max_new_tokens=settings["new_tokens"],
                                 do_sample=False)
            new = result[0, len(ids):].tolist()
            stats = lm.last
            out.append({**prompt, "input_ids": ids, "generated_ids": new,
                        "finish": getattr(stats, "finish_reason", None),
                        "top_logits": [],
                        "prefill_ms": getattr(stats, "prefill_ms", None),
                        "ms_per_token": _r(getattr(stats, "ms_per_token", None))})
    finally:
        lm.close()
    producer = {"kind": "vxpu-router", **_versions("remote"),
                "router": lm.client.address}
    return _document(model, revision, lm.config, producer, settings, out)


def _document(model, revision, config, producer, settings, prompts):
    return {"format": FORMAT,
            "model": {"repo": model, "revision": revision,
                      "model_type": getattr(config, "model_type", None)},
            "producer": producer, "settings": settings, "prompts": prompts}


def compare(reference, candidate, logit_tolerance=0.025):
    """Per prompt: tokens equal?, max top-k logit diff. Returns (rows, ok).

    ``logit_tolerance`` is relative to the largest logit at that
    position (at least 1.0): softmax is shift-invariant, so what matters
    is error against the scale of the distribution, and two bf16
    computations of the same model legitimately differ by a few units
    in the last place of that scale (a logit of 30 carries 0.25 per ulp
    in bf16). 2.5% covers that; a real divergence is tens of percent.

    The two top-k sets need not hold the same ids. An id in one set and
    not the other still has a known bound on the other side: its logit
    there is no larger than that side's k-th value, so the diff is at
    least the gap down to it. That gap is what is charged, in both
    directions; a diff of exactly the k-th-largest cut is the smallest
    the truth could be, never an acquittal.
    """
    rows, ok = [], True
    ref = {p["name"]: p for p in reference["prompts"]}
    for p in candidate["prompts"]:
        r = ref.get(p["name"])
        if r is None:
            rows.append({"prompt": p["name"], "status": "no reference"})
            continue
        same_prompt = r["input_ids"] == p["input_ids"]
        tokens_equal = r["generated_ids"] == p["generated_ids"]
        agree = sum(a == b for a, b in zip(r["generated_ids"], p["generated_ids"]))
        max_diff, max_rel = None, None
        if r.get("top_logits") and p.get("top_logits"):
            diffs, rels = [], []
            for a, b in zip(r["top_logits"], p["top_logits"]):
                av, bv = dict(zip(a["ids"], a["values"])), dict(zip(b["ids"], b["values"]))
                a_floor, b_floor = min(a["values"]), min(b["values"])
                scale = max(1.0, max(abs(v) for v in a["values"]))
                for i, v in av.items():
                    d = abs(v - bv[i]) if i in bv else max(0.0, v - b_floor)
                    diffs.append(d)
                    rels.append(d / scale)
                for i, v in bv.items():
                    if i not in av:
                        d = max(0.0, v - a_floor)
                        diffs.append(d)
                        rels.append(d / scale)
            max_diff = max(diffs) if diffs else None
            max_rel = max(rels) if rels else None
        row_ok = same_prompt and tokens_equal and (max_rel is None or max_rel <= logit_tolerance)
        ok &= row_ok
        rows.append({"prompt": p["name"], "status": "ok" if row_ok else "DIFF",
                     "same_prompt": same_prompt, "tokens_equal": tokens_equal,
                     "agree": f"{agree}/{max(len(r['generated_ids']), len(p['generated_ids']))}",
                     "max_logit_diff": None if max_diff is None else round(max_diff, 4),
                     "max_logit_rel": None if max_rel is None else round(max_rel, 4),
                     "ref_ms_per_token": r.get("ms_per_token"),
                     "ms_per_token": p.get("ms_per_token")})
    return rows, ok


def table(reference, candidate, rows):
    head = (f"{candidate['model']['repo']} @ {candidate['model']['revision']}: "
            f"{reference['producer']['kind']} vs {candidate['producer']['kind']}")
    lines = [head, "| prompt | tokens | top-k logits max diff (abs / rel) | ms/token (ref -> candidate) |",
             "|---|---|---|---|"]
    for r in rows:
        if r["status"] == "no reference":
            lines.append(f"| {r['prompt']} | no reference | | |")
            continue
        logit = ("n/a" if r["max_logit_diff"] is None
                 else f"{r['max_logit_diff']} / {100 * r['max_logit_rel']:.1f}%")
        lines.append(f"| {r['prompt']} | {'equal' if r['tokens_equal'] else 'DIFFER'} ({r['agree']}) | "
                     f"{logit} | {r['ref_ms_per_token']} -> {r['ms_per_token']} |")
    return "\n".join(lines)


def _load(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise SystemExit(f"{path}: not a {FORMAT} file")
    return doc


def _settings(args):
    return {"new_tokens": args.new_tokens, "top_k": args.top_k,
            "logit_positions": args.logit_positions, "greedy": True}


def cmd_record(args):
    settings = _settings(args)
    if args.producer == "transformers":
        doc = record_transformers(args.model, args.revision, args.prompts, settings,
                                  device=args.device, dtype=args.dtype,
                                  dequantize=args.dequantize)
    elif args.producer == "vxpu-engine":
        doc = record_vxpu_engine(args.model, args.revision, args.prompts, settings,
                                 args.artifact, device=args.device,
                                 compile_decode=args.compile)
    else:
        doc = record_vxpu_router(args.model, args.revision, args.prompts, settings,
                                 args.artifact, router=args.router)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"{args.out}: {len(doc['prompts'])} prompts by {doc['producer']['kind']}")


def cmd_compare(args):
    reference, candidate = _load(args.reference), _load(args.candidate)
    rows, ok = compare(reference, candidate, args.logit_tolerance)
    print(table(reference, candidate, rows))
    if not ok:
        raise SystemExit("vectors differ")


def selftest(workdir, settings):
    """The CI tier: tiny models through export, the in-process engine,
    and a transformers reference, token-exact. No cluster."""
    from .export import export_artifact
    from .fixtures import tiny_fp8_moe
    models = [
        ("tiny-fp8-moe", tiny_fp8_moe.write(os.path.join(workdir, "tiny-fp8-moe")), "ids"),
        ("tiny-llama", "trl-internal-testing/tiny-random-LlamaForCausalLM", "auto"),
    ]
    try:
        from .fixtures import tiny_qwen4  # hybrid linear/indexed attention, needs transformers >= 5.18
        models.append(("tiny-qwen4", tiny_qwen4.write(os.path.join(workdir, "tiny-qwen4")), "ids"))
    except ImportError as e:
        print(f"tiny-qwen4 fixture skipped: {e}")
    failed = []
    for name, model, prompts in models:
        artifact = os.path.join(workdir, name + "-artifact")
        export_artifact(model, artifact, max_cache_len=1024)
        ref = record_transformers(model, "main", prompts, settings)
        got = record_vxpu_engine(model, "main", prompts, settings, artifact)
        for label, doc in (("reference", ref), ("candidate", got)):
            with open(os.path.join(workdir, f"{name}-{label}.json"), "w") as f:
                json.dump(doc, f, indent=1)
        rows, ok = compare(ref, got)
        print(table(ref, got, rows))
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit(f"vectors differ for: {', '.join(failed)}")
    print("selftest ok:", ", ".join(m[0] for m in models))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--new-tokens", type=int, default=DEFAULTS["new_tokens"])
        p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
        p.add_argument("--logit-positions", type=int, default=DEFAULTS["logit_positions"])

    rec = sub.add_parser("record", help="record vectors with a producer")
    rec.add_argument("--producer", choices=["transformers", "vxpu-engine", "vxpu-router"], required=True)
    rec.add_argument("--model", required=True, help="Hub repo id or local checkpoint directory")
    rec.add_argument("--revision", default="main")
    rec.add_argument("--prompts", choices=["auto", "text", "ids"], default="auto",
                     help="text prompts through the chat template, or fixed ids (auto: ids if no tokenizer)")
    rec.add_argument("--artifact", help="vxpu artifact directory (vxpu-* producers)")
    rec.add_argument("--router", help="router host:port (vxpu-router; default: port-forward)")
    rec.add_argument("--device", default="cpu")
    rec.add_argument("--dtype", default="auto", help="transformers producer dtype (auto: the config's)")
    rec.add_argument("--compile", action="store_true", help="vxpu-engine: torch.compile the decode graph")
    rec.add_argument("--dequantize", action="store_true",
                     help="transformers producer: expand an fp8 checkpoint to bf16 at load")
    rec.add_argument("-o", "--out", required=True)
    common(rec)
    rec.set_defaults(func=cmd_record)

    cmp_ = sub.add_parser("compare", help="compare two vectors files")
    cmp_.add_argument("reference")
    cmp_.add_argument("candidate")
    cmp_.add_argument("--logit-tolerance", type=float, default=0.025,
                      help="relative to the position's largest logit (bf16 rounding is ~1%%)")
    cmp_.set_defaults(func=cmd_compare)

    st = sub.add_parser("selftest", help="CI tier: tiny models, in-process, token-exact")
    st.add_argument("--workdir", default=None)
    common(st)
    st.set_defaults(func=lambda a: selftest(a.workdir or tempfile.mkdtemp(prefix="vxpu-vectors-"),
                                            _settings(a)))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
