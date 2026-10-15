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
through a router) and vLLM record candidates. ``compare`` diffs two
files: where the candidate's greedy tokens leave the reference's, the
reference's logits at that position say whether the flip was a tie or
a divergence, and the top-k logits of every position up to there must
agree within a tolerance.

    python -m vxpu.vectors record --producer transformers --model REPO -o ref.json
    python -m vxpu.vectors record --producer vxpu-engine --model REPO --artifact DIR -o got.json
    python -m vxpu.vectors record --producer vxpu-router --model REPO --artifact DIR [--router host:port] -o got.json
    python -m vxpu.vectors record --producer vllm --model REPO -o vllm.json  # or: python vectors.py ...
    python -m vxpu.vectors compare ref.json got.json
    python -m vxpu.vectors selftest  # the CI tier: tiny models, in-process

Format (vxpu-vectors/1):

    {"format": "vxpu-vectors/1",
     "model": {"repo": ..., "revision": ..., "commit_sha": ...},
     "producer": {"kind": "transformers" | "vxpu-engine" | "vxpu-router" | "vllm",
                  "torch": ..., "transformers": ..., "device": ..., ...},
     "settings": {"new_tokens": 32, "top_k": 32, "logit_positions": 32,
                  "greedy": true},   # logit_positions defaults to new_tokens
     "prompts": [{"name": ..., "text": ... | "ids": [...],
                  "input_ids": [...], "generated_ids": [...],
                  "finish": "eos" | "length",
                  "top_logits": [{"ids": [...], "values": [...]}, ...],
                  "prefill_ms": ..., "ms_per_token": ...}]}

Logits are stored as the top-k (id, value) pairs of each generated
position (``logit_positions``, every generated position unless set
lower; a few tens of KB per prompt): they are what lets a token flip be judged as a tie or a
divergence, and they stay useful under any later tolerance.
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

DEFAULTS = {"new_tokens": 32, "top_k": 32, "logit_positions": None, "greedy": True}
FORMAT = "vxpu-vectors/1"


def _versions(device):
    import torch
    import transformers
    out = {"torch": torch.__version__, "transformers": transformers.__version__,
           "device": str(device)}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        # The kernel path follows from the card, so name the one used.
        out["gpu"] = torch.cuda.get_device_name(torch.device(str(device)))
    return out


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


def record_vllm(model, revision, prompts, settings, dtype="auto",
                max_model_len=None, enforce_eager=False):
    """Candidate: vLLM serving the same repo/revision, greedy, raw logits.

    vLLM is given the prompt as the same token ids the other producers
    use (the chat template is applied here, not by vLLM), and asked for
    the top-k *raw logits* of each generated position (``logprobs_mode=
    "raw_logits"``: the pre-softmax values before any processor), so the
    recording compares with the others at the logit level. Sampling
    defaults come from vLLM, not the model's generation_config, and
    everything that matters is set explicitly: temperature 0, no
    penalties. Expect larger logit differences than between two
    transformers runs: vLLM's fp8 path quantizes activations to fp8
    too (W8A8), where transformers and the vXPU graph dequantize the
    weights and multiply in bf16 (W8A16); the greedy tokens are the
    test, the logits say how close the two engines' views are.

    vLLM places the model itself (there is no device argument); the
    recording is labelled with the device and card it ran on.

    Runs with the ``vllm`` package only (plus transformers for the
    tokenizer): copy this file into a vLLM image and run it directly,
    ``python vectors.py record --producer vllm ...``.
    """
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = _tokenizer(model, revision, prompts)
    kwargs = {"model": model, "revision": revision, "dtype": dtype,
              "logprobs_mode": "raw_logits", "max_logprobs": settings["top_k"],
              "generation_config": "vllm", "enforce_eager": enforce_eager, "seed": 0}
    if max_model_len:
        kwargs["max_model_len"] = max_model_len
    llm = LLM(**kwargs)
    config = llm.llm_engine.model_config.hf_config
    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                            max_tokens=settings["new_tokens"],
                            logprobs=settings["top_k"], skip_special_tokens=False)
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        t0 = time.perf_counter()
        result = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - t0
        completion = result.outputs[0]
        new = list(completion.token_ids)
        top = []
        for step in (completion.logprobs or [])[:settings["logit_positions"]]:
            # {token_id: Logprob(logprob=<raw logit here>, rank, ...)}; the
            # sampled token is included even when outside the top-k, so
            # keep the k best by value, like torch.topk does.
            pairs = sorted(((lp.logprob, i) for i, lp in step.items()), reverse=True)
            pairs = pairs[:settings["top_k"]]
            top.append({"ids": [i for _, i in pairs],
                        "values": [round(float(v), 5) for v, _ in pairs]})
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": {"stop": "eos", "length": "length"}.get(
                        completion.finish_reason, completion.finish_reason),
                    "top_logits": top,
                    "ms_per_token": round(1000 * elapsed / max(1, len(new)), 1)})
    mc = llm.llm_engine.model_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    producer = {"kind": "vllm", "vllm": vllm.__version__, **_versions(device),
                "dtype": str(mc.dtype), "quantization": mc.quantization,
                "enforce_eager": enforce_eager, "logprobs_mode": "raw_logits"}
    return _document(model, revision, config, producer, settings, out)


def _document(model, revision, config, producer, settings, prompts):
    return {"format": FORMAT,
            "model": {"repo": model, "revision": revision,
                      "model_type": getattr(config, "model_type", None)},
            "producer": producer, "settings": settings, "prompts": prompts}


def _scale(values):
    """The scale a relative tolerance applies to at one position: the
    largest logit magnitude in the recorded top-k, at least 1.0. Used
    for logit diffs and for the tie set alike, so one tolerance means
    one thing."""
    return max(1.0, max(abs(v) for v in values))


def tie_set(top, logit_tolerance):
    """The ids a greedy decoder could defensibly pick at a position: every
    token whose logit is within the tolerance of the top one, on the
    same scale ``compare`` uses for logit diffs."""
    if not top or not top["values"]:
        return set()
    best = top["values"][0]
    allowed = logit_tolerance * _scale(top["values"])
    return {i for i, v in zip(top["ids"], top["values"]) if best - v <= allowed}


def _first_flip(r, p, logit_tolerance):
    """Where the candidate's greedy tokens leave the reference's, and
    whether the reference's logits there excuse it. None when the
    streams are equal; else a dict with the position, the two tokens,
    and a kind: ``tie`` (the candidate's token is in the reference's
    tie set), ``outside`` (it is not: the reference had a clear winner
    the candidate missed), ``unjudged`` (no reference logits at that
    position to say), or ``length`` (one side stopped where the other
    went on)."""
    pos = next((i for i, (x, y) in enumerate(zip(r["generated_ids"], p["generated_ids"]))
                if x != y), None)
    if pos is None:
        if len(r["generated_ids"]) == len(p["generated_ids"]):
            return None
        pos = min(len(r["generated_ids"]), len(p["generated_ids"]))  # one stopped early
    flip = {"position": pos,
            "ref_token": r["generated_ids"][pos] if pos < len(r["generated_ids"]) else None,
            "token": p["generated_ids"][pos] if pos < len(p["generated_ids"]) else None}
    if flip["token"] is None or flip["ref_token"] is None:
        flip["kind"] = "length"  # one side stopped (eos) where the other went on
        return flip
    tops = r.get("top_logits") or []
    if pos >= len(tops) or not tops[pos] or not tops[pos]["values"]:
        flip["kind"] = "unjudged"  # no reference logits recorded there
        return flip
    top = tops[pos]
    values = dict(zip(top["ids"], top["values"]))
    best = top["values"][0]
    flip["gap"] = round(best - values[flip["token"]], 6) if flip["token"] in values else None
    flip["kind"] = "tie" if flip["token"] in tie_set(top, logit_tolerance) else "outside"
    return flip


def compare(reference, candidate, logit_tolerance=0.025, strict_tokens=False):
    """Per prompt: do the greedy tokens hold up, and how far apart are the
    top-k logits. Returns (rows, ok).

    Greedy tokens are judged against the reference's *tie set*: at the
    first position where the candidate's token differs, the reference's
    own logits say whether that token was within the tolerance of its
    top choice. A flip inside the tie set is a coin toss two correct
    engines can call differently (eager transformers on a CPU and on an
    L4 do, at a gap of one bf16 ulp) and passes; a flip outside it is a
    divergence and fails; a flip at a position with no reference logits
    cannot be judged and fails, which is the reason to record logits
    for every generated position. ``strict_tokens=True`` (the CI
    selftest: the same computation on two paths) also fails a flip
    inside the tie set unless the logit error measured at that position
    covers the gap the flip had to close: two paths that differ by
    rounding may call an exact tie differently, but a flip across a gap
    larger than their disagreement is a bug, not rounding.

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

    Logits are compared only up to and including the first position
    where the generated tokens differ: a later position is conditioned
    on a different prefix on each side, and its logits are two
    different questions, not two answers to one.
    """
    rows, ok = [], True
    ref = {p["name"]: p for p in reference["prompts"]}
    for p in candidate["prompts"]:
        r = ref.get(p["name"])
        if r is None:
            rows.append({"prompt": p["name"], "status": "no reference"})
            continue
        same_prompt = r["input_ids"] == p["input_ids"]
        flip = _first_flip(r, p, logit_tolerance)
        tokens_equal = flip is None
        max_diff, max_rel, at_flip = None, None, None
        comparable = (len(r["generated_ids"]) if flip is None
                      else min(flip["position"] + 1, len(r["generated_ids"])))
        if r.get("top_logits") and p.get("top_logits"):
            diffs, rels = [], []
            for pos, (a, b) in enumerate(list(zip(r["top_logits"], p["top_logits"]))[:comparable]):
                av, bv = dict(zip(a["ids"], a["values"])), dict(zip(b["ids"], b["values"]))
                a_floor, b_floor = min(a["values"]), min(b["values"])
                scale = _scale(a["values"])
                here = []
                for i, v in av.items():
                    here.append(abs(v - bv[i]) if i in bv else max(0.0, v - b_floor))
                for i, v in bv.items():
                    if i not in av:
                        here.append(max(0.0, v - a_floor))
                diffs.extend(here)
                rels.extend(d / scale for d in here)
                if flip is not None and pos == flip["position"] and here:
                    at_flip = max(here)
            max_diff = max(diffs) if diffs else None
            max_rel = max(rels) if rels else None
        tokens_ok = tokens_equal or flip["kind"] == "tie"
        if strict_tokens and not tokens_equal and tokens_ok:
            # the measured error at that position must cover the gap
            flip["explained"] = (flip.get("gap") is not None and at_flip is not None
                                 and flip["gap"] <= at_flip + 1e-6)
            tokens_ok = flip["explained"]
        row_ok = same_prompt and tokens_ok and (max_rel is None or max_rel <= logit_tolerance)
        ok &= row_ok
        rows.append({"prompt": p["name"], "status": "ok" if row_ok else "DIFF",
                     "same_prompt": same_prompt, "tokens_equal": tokens_equal,
                     "tokens_ok": tokens_ok, "flip": flip,
                     "compared": f"{comparable}/{len(r['generated_ids'])}",
                     "max_logit_diff": None if max_diff is None else round(max_diff, 4),
                     "max_logit_rel": None if max_rel is None else round(max_rel, 4),
                     "ref_ms_per_token": r.get("ms_per_token"),
                     "ms_per_token": p.get("ms_per_token")})
    return rows, ok


def table(reference, candidate, rows):
    head = (f"{candidate['model']['repo']} @ {candidate['model']['revision']}: "
            f"{reference['producer']['kind']} vs {candidate['producer']['kind']}")
    lines = [head, "| prompt | tokens (compared) | top-k logits max diff (abs / rel) | ms/token (ref -> candidate) |",
             "|---|---|---|---|"]
    for r in rows:
        if r["status"] == "no reference":
            lines.append(f"| {r['prompt']} | no reference | | |")
            continue
        logit = ("n/a" if r["max_logit_diff"] is None
                 else f"{r['max_logit_diff']} / {100 * r['max_logit_rel']:.1f}%")
        lines.append(f"| {r['prompt']} | {_describe_tokens(r)} ({r['compared']}) | "
                     f"{logit} | {r['ref_ms_per_token']} -> {r['ms_per_token']} |")
    return "\n".join(lines)


def _describe_tokens(row):
    f = row["flip"]
    if f is None:
        return "equal"
    where = f"@{f['position']}"
    if f["kind"] == "tie":
        return f"tie flip {where}, gap {f['gap']}"
    if f["kind"] == "outside":
        return f"DIFFER {where}, gap {f['gap'] if f['gap'] is not None else 'beyond top-k'}"
    if f["kind"] == "length":
        return f"DIFFER {where}, {'candidate' if f['token'] is None else 'reference'} stopped"
    return f"DIFFER {where}, no logits to judge"


def _load(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise SystemExit(f"{path}: not a {FORMAT} file")
    return doc


def _settings(args):
    # Logits for every generated position unless asked for fewer: a flip
    # at a position without them cannot be judged.
    positions = args.logit_positions if args.logit_positions is not None else args.new_tokens
    return {"new_tokens": args.new_tokens, "top_k": args.top_k,
            "logit_positions": positions, "greedy": True}


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
    elif args.producer == "vllm":
        doc = record_vllm(args.model, args.revision, args.prompts, settings,
                          dtype=args.dtype,
                          max_model_len=args.max_model_len,
                          enforce_eager=args.enforce_eager)
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


def _check_compare():
    """compare's rules on synthetic vectors: an exact tie the candidate
    calls the other way passes strict mode when the measured logit
    error covers the gap, and fails it when the gap is wider than the
    paths' disagreement."""
    def doc(kind, ids, tops):
        return {"producer": {"kind": kind},
                "prompts": [{"name": "p", "input_ids": [1, 2], "generated_ids": ids,
                             "top_logits": [{"ids": list(t), "values": list(v)} for t, v in tops]}]}
    ref = doc("a", [7, 3], [((7, 3, 5), (10.0, 10.0, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    # same logits to within 0.001 at the tie, other token picked: rounding
    got = doc("b", [3, 3], [((3, 7, 5), (10.0005, 10.0, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    _, ok = compare(ref, got, strict_tokens=True)
    assert ok, "a flip at an exact tie within the measured error must pass strict mode"
    # the gap (0.2) is wider than any logit disagreement (0.001): not rounding
    ref2 = doc("a", [7, 3], [((7, 3, 5), (10.0, 9.8, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    got2 = doc("b", [3, 3], [((7, 3, 5), (10.001, 9.8, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    rows, ok = compare(ref2, got2, strict_tokens=True)
    assert not ok and rows[0]["flip"]["kind"] == "tie" and rows[0]["flip"]["explained"] is False
    _, ok = compare(ref2, got2)  # the cluster tier: within the tie tolerance, passes
    assert ok


def selftest(workdir, settings):
    """The CI tier: tiny models through export, the in-process engine,
    and a transformers reference, token-exact. No cluster."""
    _check_compare()
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
        rows, ok = compare(ref, got, strict_tokens=True)
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
        p.add_argument("--logit-positions", type=int, default=None,
                       help="positions to record top-k logits for (default: all of --new-tokens)")

    rec = sub.add_parser("record", help="record vectors with a producer")
    rec.add_argument("--producer", choices=["transformers", "vxpu-engine", "vxpu-router", "vllm"], required=True)
    rec.add_argument("--model", required=True, help="Hub repo id or local checkpoint directory")
    rec.add_argument("--revision", default="main")
    rec.add_argument("--prompts", choices=["auto", "text", "ids"], default="auto",
                     help="text prompts through the chat template, or fixed ids (auto: ids if no tokenizer)")
    rec.add_argument("--artifact", help="vxpu artifact directory (vxpu-* producers)")
    rec.add_argument("--router", help="router host:port (vxpu-router; default: port-forward)")
    rec.add_argument("--device", default="cpu", help="transformers / vxpu-engine producers (vllm places itself)")
    rec.add_argument("--dtype", default="auto", help="transformers / vllm model dtype (auto: the config's)")
    rec.add_argument("--compile", action="store_true", help="vxpu-engine: torch.compile the decode graph")
    rec.add_argument("--dequantize", action="store_true",
                     help="transformers producer: expand an fp8 checkpoint to bf16 at load")
    rec.add_argument("--max-model-len", type=int, default=None, help="vllm: context length to allocate for")
    rec.add_argument("--enforce-eager", action="store_true",
                     help="vllm: no CUDA graphs / torch.compile")
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
