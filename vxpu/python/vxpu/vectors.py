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
import sys
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


def _progress(prompt, new, ms_per_token):
    """One line per finished prompt on stderr, so a long run (the real
    DeepSeek reference takes hours on a CPU) shows where it is."""
    rate = f"{ms_per_token:.0f} ms/token" if ms_per_token else "no timing"
    print(f"[vectors] {prompt['name']}: {len(new)} tokens, {rate}", file=sys.stderr, flush=True)


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
        _progress(prompt, new, 1000 * elapsed / max(1, len(new)))
    producer = {"kind": "transformers", **_versions(lm.device), "dtype": str(lm.dtype),
                "dequantize": dequantize,
                "experts_implementation": getattr(lm.config, "_experts_implementation", None)}
    return _document(model, revision, lm.config, producer, settings, out)


def _download(url, dest, size=None):
    """Stream url to dest (resumes nothing; a complete file of the
    expected size is kept). Sends the Hub token when one is configured."""
    import requests
    from huggingface_hub.utils import build_hf_headers
    if size is not None and os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    with requests.get(url, headers=build_hf_headers(), stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    os.replace(part, dest)
    return dest


def record_deepseek_eager(model, revision, prompts, settings):
    """Reference for a deepseek_v41 checkpoint: the model maker's module
    from the fork, run eagerly with the plain-aten kernels on the CPU (no
    transformers implementation exists; the plain-aten kernels are CPU
    code, so there is no device to choose). Greedy; logits at every step."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from . import deepseek_v41 as ds
    from .manifest import build_manifest
    manifest = build_manifest(model, revision)
    tok = _tokenizer(model, revision, prompts)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    fork = ds._fork()
    args = ds.model_args(model, settings.get("max_cache_len", 1024))
    with ds.bf16_default():
        lm = fork.Transformer(args, tokenizer).eval()
    # Load shard by shard into the module: the checkpoint never coexists
    # with the parameters in memory, so a 510 GB model needs ~510 GB, not
    # twice that. Remote shards are streamed a few at a time ahead of the
    # loader and deleted once loaded (hf_hub_download refuses files over
    # 50 GB without its xet client, and the engram shards are ~100 GB).
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_url
    cache = os.path.join(os.environ.get("VXPU_SHARD_CACHE", os.path.join(tempfile.gettempdir(), "vxpu-shards")),
                         model.replace("/", "--"))

    def fetch(ref):
        if ref["source"].startswith("file://"):
            return ref["source"][len("file://"):], False
        return _download(hf_hub_url(model, ref["name"], revision=revision),
                         os.path.join(cache, ref["name"]), ref.get("size")), True

    # Every tensor the module declares must come from some shard, else the
    # reference would run on random initialisation and the comparator would
    # blame the engine. A key is missing only if every shard load reported
    # it so; expert banks are assembled from per-expert checkpoint tensors
    # (the module's _load_from_state_dict), so they are checked by count.
    import re
    expert_re = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
    missing_all, experts_seen = None, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, ref) for ref in sorted(manifest["files"].values(), key=lambda r: r["name"])]
        for fut in futures:
            path, fetched = fut.result()
            shard = load_file(path)
            for key in shard:
                m = expert_re.match(key)
                if m:
                    experts_seen.setdefault((int(m.group(1)), m.group(3), m.group(4)), set()).add(int(m.group(2)))
            missing, unexpected = lm.load_state_dict(shard, strict=False)
            missing_all = set(missing) if missing_all is None else missing_all & set(missing)
            unexpected = [u for u in unexpected if not u.startswith(("mtp.", "vision.", "aligner.", "image_"))]
            if unexpected:
                raise RuntimeError(f"checkpoint tensors the module did not take: {unexpected[:5]}")
            del shard
            if fetched:
                os.remove(path)
    bank_re = re.compile(r"^layers\.(\d+)\.ffn\.(w[123])(_scale)?$")
    problems = sorted(k for k in (missing_all or set()) if not bank_re.match(k))
    for key in sorted(k for k in (missing_all or set()) if bank_re.match(k)):
        layer, proj, scale = bank_re.match(key).groups()
        seen = experts_seen.get((int(layer), proj, "scale" if scale else "weight"), set())
        if len(seen) != args.n_routed_experts:
            problems.append(f"{key}: {len(seen)} of {args.n_routed_experts} experts in the checkpoint")
    if problems:
        raise RuntimeError(f"module tensors the checkpoint did not provide: {problems[:5]}")
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        t0 = time.perf_counter()
        new, top = [], []
        # The exporter sets the expert routing per graph (grouped for the
        # prefill, batched for the decode step); the module's default is
        # grouped, which on a decode step dequantizes every expert of
        # every layer (384 x 40 here) for one token's six. Same math,
        # ~40x the work: nine hours instead of fifteen minutes.
        with torch.no_grad(), ds.bf16_default():
            ds.set_routing(lm, "grouped")
            logits = lm(torch.tensor([ids]), 0)[1]
            ds.set_routing(lm, "batched")
            for step in range(settings["new_tokens"]):
                if step < settings["logit_positions"]:
                    top.append(_top(logits[0], settings["top_k"]))
                nxt = int(logits[0].argmax())
                new.append(nxt)
                if nxt in eos_ids:
                    break
                logits = lm(torch.tensor([[nxt]]), len(ids) + step)[1]
        elapsed = time.perf_counter() - t0
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": "eos" if new and new[-1] in eos_ids else "length",
                    "top_logits": top,
                    "ms_per_token": round(1000 * elapsed / max(1, len(new)), 1)})
        _progress(prompt, new, 1000 * elapsed / max(1, len(new)))
    producer = {"kind": "deepseek-eager", **_versions("cpu"), "kernels": "aten"}
    return _document(model, revision, type("Config", (), {"model_type": ds.MODEL_TYPE})(), producer, settings, out)


def record_vxpu_engine(model, revision, prompts, settings, artifact, device="cpu",
                       compile_decode=False, telemetry=False, keep=None):
    """Candidate: the executor's engine in this process (no cluster).
    telemetry=True keeps each prompt's execution telemetry with it; keep,
    a list, receives the engine (for checks of its state afterwards)."""
    import torch
    from .engine import Engine, SamplingParams
    tok = _tokenizer(model, revision, prompts)
    engine = Engine(artifact, device=device, compile_decode=compile_decode)
    if keep is not None:
        keep.append(engine)
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
                                     logits_sink=sink, telemetry=telemetry):
            new.extend(event.get("token_ids", []))
            if event.get("done"):
                stats = event
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": stats.get("finish_reason") if stats else None,
                    "top_logits": top,
                    "prefill_ms": stats.get("prefill_ms") if stats else None,
                    "ms_per_token": _r(stats.get("ms_per_token")) if stats else None,
                    **({"telemetry": stats.get("telemetry")} if telemetry and stats else {})})
        _progress(prompt, new, stats.get("ms_per_token") if stats else None)
    with open(os.path.join(artifact, "binding.json")) as f:
        binding = json.load(f)
    from .plan import record as plan_record
    producer = {"kind": "vxpu-engine", **_versions(device),
                "compile": compile_decode,
                "binding": {k: len(binding.get(k, [])) for k in
                            ("bound", "derived", "state", "stacked", "aliases")},
                "quantization": binding.get("quantization"),
                "plan": plan_record(engine.explain())}
    return _document(model, revision, engine.config, producer, settings, out)



def record_vxpu_router(model, revision, prompts, settings, artifact, router=None, telemetry=False):
    """Candidate: an executor reached through a router (tokens only;
    the wire carries no logits). telemetry=True keeps each prompt's
    execution telemetry, which rides back on the Generate stream."""
    from .modeling import VxpuModelForCausalLM
    tok = _tokenizer(model, revision, prompts)
    lm = VxpuModelForCausalLM.from_pretrained(
        model, router=router, artifact_dir=artifact, revision=revision,
        progress=lambda *a: None)
    lm.collect_telemetry = telemetry
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
                        "ms_per_token": _r(getattr(stats, "ms_per_token", None)),
                        **({"telemetry": getattr(stats, "telemetry", None)} if telemetry else {})})
            _progress(prompt, new, getattr(stats, "ms_per_token", None))
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
        _progress(prompt, new, 1000 * elapsed / max(1, len(new)))
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
                                 compile_decode=args.compile, telemetry=args.telemetry)
    elif args.producer == "deepseek-eager":
        doc = record_deepseek_eager(args.model, args.revision, args.prompts, settings)
    elif args.producer == "vllm":
        doc = record_vllm(args.model, args.revision, args.prompts, settings,
                          dtype=args.dtype,
                          max_model_len=args.max_model_len,
                          enforce_eager=args.enforce_eager)
    else:
        doc = record_vxpu_router(args.model, args.revision, args.prompts, settings,
                                 args.artifact, router=args.router, telemetry=args.telemetry)
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


def _check_compile_plan():
    """Verify configure_inductor and Plan's compiler choices for prologue_fusion."""
    from . import engine, plan

    # CPU: prologue_fusion is False, marked CUDA only
    p_cpu = plan.Plan("cpu", compile_decode=False)
    assert p_cpu["prologue_fusion"] is False
    assert p_cpu.choices["prologue_fusion"].source == "CUDA only"

    # CUDA without compile_decode: requires compile_decode
    p_cuda_no_compile = plan.Plan("cuda", compile_decode=False)
    assert p_cuda_no_compile["prologue_fusion"] is False
    assert p_cuda_no_compile.choices["prologue_fusion"].source == "requires compile_decode"

    # CUDA with compile_decode: True by default
    p_cuda_compile = plan.Plan("cuda", compile_decode=True)
    assert p_cuda_compile["prologue_fusion"] is True
    assert p_cuda_compile.choices["prologue_fusion"].source == "default"

    # Recorded in explain choices
    explained = p_cuda_compile.explain()
    assert "prologue_fusion" in explained["choices"]
    assert explained["choices"]["prologue_fusion"]["value"] is True

    # configure_inductor sets torch._inductor.config.prologue_fusion when enabled by the plan
    try:
        import torch._inductor.config as inductor_config
        orig = getattr(inductor_config, "prologue_fusion", None)
        try:
            inductor_config.prologue_fusion = False
            engine.configure_inductor(p_cpu)
            assert inductor_config.prologue_fusion is False  # untouched on CPU
            engine.configure_inductor(p_cuda_compile)
            assert inductor_config.prologue_fusion is True
            engine.configure_inductor(p_cpu)
            assert inductor_config.prologue_fusion is True  # left untouched
        finally:
            if orig is not None:
                inductor_config.prologue_fusion = orig
    except ImportError:
        pass


def _check_expert_sim():
    """expert_sim's policies on synthetic traffic whose hot set moves: four
    requests use experts 0-3, then eight use 4-7, a budget of four. The
    static placement (from the first four) serves none of the later ones;
    holding the last request's experts (decay 0) misses only the first;
    cumulative LFU (decay 1) switches once the new set's counts pass the
    old; the prefill (here the same experts as the decode) and the oracle
    serve them all."""
    from . import expert_sim
    import collections
    def req(experts):
        c = collections.Counter({(0, e): 10 for e in experts})
        return c, 10, c
    reqs = [req(range(4))] * 4 + [req(range(4, 8))] * 8
    def share(policy, **kw):
        got = expert_sim.simulate(reqs, 4, 4, policy, **kw)
        return sum(s for s, _ in got) / len(got), sum(m for _, m in got)
    assert share("static") == (0.0, 0), share("static")
    assert share("adaptive", decay=0.0) == (7 / 8, 4), share("adaptive", decay=0.0)
    assert share("adaptive", decay=1.0) == (3 / 8, 4), share("adaptive", decay=1.0)
    assert share("prefill", prefill_weight=0.75) == (1.0, 4), share("prefill", prefill_weight=0.75)
    assert share("oracle") == (1.0, 4), share("oracle")


def _check_continuation(artifact, ids, new_tokens=4):
    """A session extended by a later turn must see what a fresh session
    given the whole transcript sees. For a model whose prefill graph
    starts at position 0 the engine re-prefills from the start, so the
    two are bit-identical; this is the multi-turn case of chat."""
    import torch
    from .engine import Engine, SamplingParams
    engine = Engine(artifact)

    def run(sid, ids):
        first = {}

        def sink(position, logits):
            if position == 0:
                first["logits"] = logits.detach().float().clone()
        gen = []
        for ev in engine.generate(sid, ids, new_tokens, SamplingParams(do_sample=False), logits_sink=sink):
            gen.extend(ev.get("token_ids", []))
        return gen, first["logits"]

    sid = engine.new_session()
    gen1, _ = run(sid, ids)
    turn2 = ids + gen1 + ids[:7]  # the transcript so far plus a new turn
    gen2, extended = run(sid, turn2)
    gen3, fresh = run(engine.new_session(), turn2)
    if gen2 != gen3 or not torch.equal(extended, fresh):
        raise SystemExit(f"a continued session differs from a fresh one: {gen2} vs {gen3}, "
                         f"max logit diff {(extended - fresh).abs().max().item():.4g}")
    print(f"  continuation ok: {len(turn2)}-token second turn matches a fresh session")


def _check_telemetry(model, prompts, settings, artifact, plain, workdir):
    """The engine's execution telemetry, end to end: recording it changes
    no token; each prompt carries a vxpu-telemetry/1 record with its
    prefilled token count and decode expert counts that are the same
    multiple of the decode steps in every layer (each step routes one
    token to top-k experts per layer); and expert_stats reads the file."""
    from . import expert_stats
    doc = record_vxpu_engine(model, "main", prompts, settings, artifact, telemetry=True)
    path = os.path.join(workdir, "tiny-deepseek-v41-telemetry.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    problems = []
    for a, b in zip(plain["prompts"], doc["prompts"]):
        tel = b.get("telemetry") or {}
        if a["generated_ids"] != b["generated_ids"]:
            problems.append(f"{b['name']}: tokens changed with telemetry on")
        if tel.get("format") != "vxpu-telemetry/1":
            problems.append(f"{b['name']}: no telemetry record")
            continue
        if tel.get("prefill_tokens") != len(b["input_ids"]):  # a fresh session prefills the whole prompt
            problems.append(f"{b['name']}: prefill_tokens {tel.get('prefill_tokens')} for {len(b['input_ids'])} prompt tokens")
        steps = tel["decode_steps"]
        totals = {sum(c.values()) for c in tel.get("experts", {}).get("decode", {}).values()}
        if not steps or len(totals) != 1 or next(iter(totals)) % steps:
            problems.append(f"{b['name']}: decode expert counts {sorted(totals)} for {steps} decode steps")
    _, text = expert_stats.report([path])
    print(text)
    for p in problems:
        print("telemetry check:", p)
    return not problems


def _check_lowering(model, prompts, settings, artifact, workdir):
    """On a GPU: the compiled, split decode with the plan lowered into its
    segments (lower.py), and with those segments as AOTInductor packages
    in the engine's own CUDA graphs (aoti.py), each against the plain
    compiled decode, token-exact. Lowered calls run the same kernels, so
    the vectors must agree. Each variant must really have run as such
    (the plan falls back silently when a requirement is missing). Skipped
    without CUDA or Triton (nothing is lowerable)."""
    import torch
    if not torch.cuda.is_available():
        print("lowering check skipped: no CUDA")
        return True
    # every switch the variants depend on is pinned, so none is inherited
    # from the environment the selftest runs in
    split = {"VXPU_SPLIT_DECODE": "1", "VXPU_SPLIT_COMPILE": "1"}
    variants = {"compiled": {**split, "VXPU_SEGMENT_RUNTIME": "jit", "VXPU_LOWER_DECODE": "0"},
                "lowered": {**split, "VXPU_SEGMENT_RUNTIME": "jit", "VXPU_LOWER_DECODE": "1"},
                "aoti": {**split, "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_LOWER_DECODE": "1",
                         "VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "aoti-cache")}}
    intended = {"compiled": (False, "jit"), "lowered": (True, "jit"), "aoti": (True, "aoti")}  # lower_decode, runtime
    runs = {}
    for label, env in variants.items():
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            runs[label] = record_vxpu_engine(model, "main", prompts, settings, artifact, device="cuda",
                                             compile_decode=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k)
                else:
                    os.environ[k] = v
    from .lower import REPORT_PREFIX
    if not runs["lowered"]["producer"]["plan"]["choices"].get("triton"):
        print("lowering check skipped: no Triton (nothing is lowerable)")
        return True
    ok = True
    for label, (lowered, runtime) in intended.items():  # each run must have run as intended
        choices = runs[label]["producer"]["plan"]["choices"]
        got = (bool(choices.get("lower_decode")), choices.get("segment_runtime"))
        if got != (lowered, runtime):
            print(f"lowering check: the {label} run ran lower_decode={got[0]}, segment_runtime={got[1]}")
            ok = False
    if not ok:
        return False
    for label in ("lowered", "aoti"):
        plan = runs[label]["producer"]["plan"]
        report = [n for n in plan.get("notes", []) if n.startswith(REPORT_PREFIX)]
        if not report or "lowered " not in report[0][len(REPORT_PREFIX):]:
            print(f"lowering check: the {label} run lowered nothing (report={report})")
            ok = False
            continue
        rows, same = compare(runs["compiled"], runs[label], strict_tokens=True)
        print(f"{label} decode against compiled, cuda:")
        print(table(runs["compiled"], runs[label], rows))
        ok = ok and same
    return ok

def _check_host_reads(workdir):
    """The n-gram tables' reads (#364): the CPU kernels' gather_rows gives
    index_select's rows and refuses indices out of range or not on the
    CPU (it reads them on the host); advise_random takes on a
    memory-mapped tensor."""
    import mmap
    import torch
    from . import kernels
    from .rehydrate import advise_random, file_backed
    k = kernels.load()
    if k is None or not hasattr(k, "gather_rows"):
        print("host reads check skipped: no CPU kernels")
        return
    table = torch.arange(64 * 24, dtype=torch.int16).view(64, 24)
    idx = torch.tensor([[3, 63, 0], [17, 17, 5]])
    if not torch.equal(k.gather_rows(table, idx), table[idx]):
        raise SystemExit("gather_rows: rows differ from index_select's")
    for bad in (torch.tensor([64]), torch.tensor([-1])):
        try:
            k.gather_rows(table, bad)
        except RuntimeError as e:
            if "outside" not in str(e):
                raise
        else:
            raise SystemExit(f"gather_rows accepted index {bad.item()} for a 64-row table")
    device = "cuda" if torch.cuda.is_available() else "meta"
    try:
        k.gather_rows(table, idx.to(device))
    except RuntimeError as e:
        if "indices on the CPU" not in str(e):
            raise
    else:
        raise SystemExit(f"gather_rows accepted indices on {device}")
    path = os.path.join(workdir, "advise-random.bin")
    mapped, _ = file_backed(path, 1 << 16)
    if getattr(mmap, "MADV_RANDOM", None) is not None and not advise_random(mapped):
        raise SystemExit("advise_random: madvise(MADV_RANDOM) was refused on a memory-mapped tensor")
    print("host reads: gather_rows matches index_select and refuses bad or device indices; MADV_RANDOM taken")


def _check_local_blobs_copy(workdir):
    """Copying into the local weight tier (localblobs.py) through a fake
    gcloud that copies every file but one and then fails, as gcloud does:
    the complete files are kept, the next copy fetches only the missing
    one, and staging left by a process that no longer runs is removed."""
    import subprocess
    from . import localblobs
    root = os.path.join(workdir, "blob-copy")
    bucket_dir, blob_dir, bindir = (os.path.join(root, d) for d in ("bucket/_blobs", "blobs", "bin"))
    for d in (bucket_dir, blob_dir, bindir):
        os.makedirs(d, exist_ok=True)
    files = {}
    for k in range(3):
        sha = f"{k:064x}"
        with open(os.path.join(bucket_dir, sha), "wb") as f:
            f.write(bytes([k]) * (1000 + k))
        files[sha] = {"size": 1000 + k, "name": f"part{k}", "source": "unused"}
    fake = os.path.join(bindir, "gcloud")
    with open(fake, "w") as f:
        f.write(f"""#!{sys.executable}
import os, shutil, sys
dest, skip, failed = sys.argv[-1], os.environ.get("FAKE_GCLOUD_SKIP"), False
for url in sys.stdin.read().split():
    sha = url.rsplit("/", 1)[1]
    if sha == skip:
        failed = True
        continue
    shutil.copyfile(os.path.join({os.path.dirname(bucket_dir)!r}, "_blobs", sha), os.path.join(dest, sha))
sys.exit(1 if failed else 0)
""")
    os.chmod(fake, 0o755)
    gone = subprocess.Popen(["true"])
    gone.wait()
    stale = os.path.join(blob_dir, ".incoming", f"{gone.pid}-stale")
    os.makedirs(stale)
    skip = next(iter(files))
    env = {"PATH": bindir + os.pathsep + os.environ.get("PATH", ""), "FAKE_GCLOUD_SKIP": skip}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        try:
            localblobs.localize(files, blob_dir, bucket="gs://fake", log=lambda *_: None)
        except RuntimeError as e:
            if "2 of 3 files copied and kept" not in str(e):
                raise
        else:
            raise SystemExit("local blobs: a failed gcloud copy was not reported")
        kept = sorted(n for n in os.listdir(blob_dir) if not n.startswith("."))
        if kept != sorted(set(files) - {skip}) or os.listdir(os.path.join(blob_dir, ".incoming")):
            raise SystemExit(f"local blobs: after a partial copy, kept {kept}, staging "
                             f"{os.listdir(os.path.join(blob_dir, '.incoming'))}")
        os.environ.pop("FAKE_GCLOUD_SKIP")
        _, summary = localblobs.localize(files, blob_dir, bucket="gs://fake", log=lambda *_: None)
        if summary["copied"] != 1 or summary["local"] != 3:
            raise SystemExit(f"local blobs: the retry should copy only the missing file: {summary}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("local blobs copy: a partial copy keeps its complete files, the retry copies the rest, stale staging swept")


def _check_local_blobs(model, prompts, settings, artifact, plain, workdir):
    """The executor machine's local weight tier (localblobs.py): with the
    checkpoint's files in a blob directory, named by sha256, every source
    points there and the engine's tokens are unchanged."""
    import shutil
    from . import localblobs
    with open(os.path.join(artifact, "manifest.json")) as f:
        files = json.load(f)["files"]
    blob_dir = os.path.join(workdir, "blobs")
    os.makedirs(blob_dir, exist_ok=True)
    for sha, entry in files.items():
        shutil.copyfile(entry["source"][len("file://"):], os.path.join(blob_dir, sha))
    local, summary = localblobs.localize(files, blob_dir)
    if summary["local"] != len(files) or any(not e["source"].startswith("file://" + os.path.abspath(blob_dir))
                                             for e in local.values()):
        print(f"local blobs check: sources not localized: {summary}")
        return False
    saved = os.environ.get("VXPU_BLOB_DIR")
    os.environ["VXPU_BLOB_DIR"] = blob_dir
    try:
        got = record_vxpu_engine(model, "main", prompts, settings, artifact)
    finally:
        if saved is None:
            os.environ.pop("VXPU_BLOB_DIR")
        else:
            os.environ["VXPU_BLOB_DIR"] = saved
    if [p["generated_ids"] for p in got["prompts"]] != [p["generated_ids"] for p in plain["prompts"]]:
        print("local blobs check: tokens changed when reading from the local weight tier")
        return False
    print(f"local weight tier: {summary['local']} files read from {blob_dir}, tokens unchanged")
    return True


def _check_compile_names():
    """Names a compile writes (compile.py, modelstore.py, manifest.py): a tag
    must match modelstore's grammar as a whole string (a trailing newline
    passes a $-anchored match, then modelstore refuses it after the
    export), and a file's recorded source is the Hub's only for a commit."""
    from . import manifest, modelstore
    for good in ("main", "314113f8a7da_hero1", "v1.2-x"):
        if not modelstore.TAG_RE.fullmatch(good):
            raise SystemExit(f"tag {good!r} refused")
    for bad in ("v1\n", "-v1", "a/b", "", "x" * 129):
        if modelstore.TAG_RE.fullmatch(bad):
            raise SystemExit(f"tag {bad!r} accepted")
    commit = "314113f8a7dadb388215396ba5fbee2c610a04d6"
    src, read = manifest._hub_source("org/m", commit, "a.safetensors", "http://modelstore")
    if src != f"https://huggingface.co/org/m/resolve/{commit}/a.safetensors" or not read.startswith("http://modelstore/"):
        raise SystemExit(f"a commit's source should be the Hub's: {src}")
    src, read = manifest._hub_source("org/m", "main", "a.safetensors", "http://modelstore")
    if src != read:
        raise SystemExit(f"a moving ref's source should be the endpoint read: {src}")


def _check_chat_telemetry(artifact):
    """Chat (text in, text out; what clients without a tokenizer use)
    returns the turn's execution telemetry when asked, as generate does;
    the router records it for models served by name."""
    from .engine import Engine
    engine = Engine(artifact)
    reply = engine.chat(engine.new_session(), "Is the sky blue?", max_new_tokens=4, telemetry=True)
    tel = reply.get("telemetry") or {}
    if tel.get("format") != "vxpu-telemetry/1" or tel.get("decode_steps", 0) < 1:
        print(f"chat telemetry check: no telemetry record in the reply ({sorted(reply)})")
        return False
    plain = engine.chat(engine.new_session(), "Is the sky blue?", max_new_tokens=4)
    if "telemetry" in plain or plain["text"] != reply["text"]:
        print("chat telemetry check: telemetry changed the reply, or was recorded unasked")
        return False
    print(f"chat telemetry: {tel['decode_steps']} decode steps recorded, reply unchanged")
    return True


def _check_gpu_experts(model, prompts, settings, artifact, workdir):
    """On a GPU: the hot experts' kernels (triton_ops.fp4_experts_decode)
    against the module's own arithmetic on random fp4 banks, then the tiny
    DeepSeek with half of every layer's experts placed on the GPUs (split
    between two where there are two; experts_gpu.py) against the same
    model computing them all on the CPU (a different summation order: the rounding class, so a near tie
    may flip). Skipped without CUDA and Triton."""
    import torch
    from . import triton_ops
    if not triton_ops.available():
        print("gpu experts check skipped: no CUDA with Triton")
        return True
    from .deepseek_v41 import _fork
    ds = _fork()
    K, I, S, dev = 256, 128, 4, "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    def bank(n_out, n_in):
        w = torch.randint(0, 256, (S, n_out, n_in // 2), dtype=torch.uint8, device=dev, generator=g)
        sc = torch.randint(120, 125, (S, n_out, n_in // 32), dtype=torch.uint8, device=dev, generator=g)
        return w, sc.view(torch.float8_e8m0fnu)
    (w1, s1), (w3, s3), (w2, s2) = bank(I, K), bank(I, K), bank(K, I)
    x = triton_ops.act_quant_dequant(torch.randn(1, K, device=dev, generator=g).to(torch.bfloat16))
    weights = torch.rand(1, 3, device=dev, generator=g)
    slots = torch.tensor([[2, -1, 0]], device=dev)
    y = triton_ops.fp4_experts_decode(x, weights, slots, w1, s1, w3, s3, w2, s2, 10.0)

    class _M:
        swiglu_limit = 10.0
        def _quantized_input(self, t):
            return ds.MoE._quantized_input(self, t)
    ref = torch.zeros(1, K, device=dev)
    for k, sl in enumerate(slots[0].tolist()):
        if sl < 0:
            continue
        deq = [ds.MoE._dequant(w[sl:sl + 1], sc[sl:sl + 1]) for w, sc in ((w1, s1), (w3, s3), (w2, s2))]
        gate, up = (torch.bmm(m, x.view(1, K, 1)).squeeze(-1) for m in deq[:2])
        h = ds.MoE._ffn(_M(), gate, up, weights[0, k:k + 1])
        ref[0] += torch.bmm(deq[2], h.unsqueeze(-1)).squeeze(-1).float()[0]
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    if rel > 1e-2:
        print(f"gpu experts check: the kernels differ from the module's arithmetic (relative {rel:.3g})")
        return False

    with open(os.path.join(artifact, "binding.json")) as f:
        experts = (json.load(f).get("external") or {}).get("experts") or {}
    if not experts:
        print("gpu experts check: the model places no experts externally; kernels checked only")
        return True
    n = len(next(iter(next(iter(experts.values())).values()))["parts"])
    # half of every layer's experts on the GPUs: with two, alternating
    # between them, so every layer has pools on both
    gpus = min(torch.cuda.device_count(), 2)
    held = list(range(0, n, 2))
    placement = {"format": "vxpu-placement/2",
                 "gpus": [{"layers": {layer: held[g::gpus] for layer in experts}} for g in range(gpus)]}
    path = os.path.join(workdir, "placement.json")
    with open(path, "w") as f:
        json.dump(placement, f)
    runs = {}
    for label, value in (("cpu", None), ("gpu", path)):
        saved = os.environ.get("VXPU_GPU_EXPERTS")
        if value is None:
            os.environ.pop("VXPU_GPU_EXPERTS", None)
        else:
            os.environ["VXPU_GPU_EXPERTS"] = value
        try:
            runs[label] = record_vxpu_engine(model, "main", prompts, settings, artifact, device="cuda")
        finally:
            if saved is None:
                os.environ.pop("VXPU_GPU_EXPERTS", None)
            else:
                os.environ["VXPU_GPU_EXPERTS"] = saved
    if "gpu-hot+cpu" not in runs["gpu"]["producer"]["plan"]["selected"].get("vxpu::moe_experts", []):
        print("gpu experts check: the hot experts' implementation never ran")
    runs["cpu"] = _with_env({"VXPU_GPU_EXPERTS": None}, record_vxpu_engine,
                            model, "main", prompts, settings, artifact, device="cuda")
    runs["gpu"] = _with_env({"VXPU_GPU_EXPERTS": path}, record_vxpu_engine,
                            model, "main", prompts, settings, artifact, device="cuda", telemetry=True)
    if not any(n.startswith("hot experts on") for n in runs["gpu"]["producer"]["plan"].get("notes", [])):
        print("gpu experts check: the placement was not applied")
        return False
    rows, ok = compare(runs["cpu"], runs["gpu"], strict_tokens=True)
    print(f"hot experts on {gpus} GPU(s) (kernels relative error {rel:.2g}) against all on the CPU:")
    print(table(runs["cpu"], runs["gpu"], rows))
    return _check_expert_moves(model, prompts, settings, artifact, workdir, runs, gpus, list(experts), n) and ok


def _check_expert_moves(model, prompts, settings, artifact, workdir, runs, gpus, layers, n):
    """Experts moving with the traffic (experts_gpu.Mover): start from the
    worst placement, the experts the previous run selected least, so moves
    must happen; then the tokens against all on the CPU, and every held
    expert's rows on its GPU against the host's (experts_gpu.verify)."""
    import torch
    from . import experts_gpu
    from .expert_stats import collect
    counts = collect([runs["gpu"]])["decode"]
    cold = {layer: sorted(range(n), key=lambda e: counts[int(layer)][e])[:n // 2] for layer in layers}
    placement = {"format": "vxpu-placement/2",
                 "gpus": [{"layers": {layer: ids[g::gpus] for layer, ids in cold.items()}} for g in range(gpus)]}
    path = os.path.join(workdir, "placement-cold.json")
    with open(path, "w") as f:
        json.dump(placement, f)
    keep = []
    moved = _with_env({"VXPU_GPU_EXPERTS": path, "VXPU_GPU_EXPERTS_MOVES": "2"},
                      record_vxpu_engine, model, "main", prompts, settings, artifact, device="cuda",
                      telemetry=True, keep=keep)
    mover = keep[0].expert_mover
    recorded = sum(p["telemetry"]["host_ops"].get("decode", {}).get("moe_experts", {}).get("moves_completed", 0.0)
                   for p in moved["prompts"])
    if not (mover.completed and recorded):
        print(f"gpu experts check: from a placement of the least selected, {mover.started} moves started, "
              f"{mover.completed} completed ({recorded:g} per step in the telemetry)")
        return False
    torch.cuda.synchronize()
    for moe in keep[0].external.experts.values():
        res = getattr(moe, "_vxpu_hot", None)
        if res is not None:
            experts_gpu.verify(res)
    rows, ok = compare(runs["cpu"], moved, strict_tokens=True)
    print(f"experts moving with the traffic ({mover.started} moves started, {mover.completed} completed; "
          "the residency matches the host banks) against all on the CPU:")
    print(table(runs["cpu"], moved, rows))
    return ok


def _check_expert_mailboxes(model, prompts, settings, artifact, plain, workdir):
    """Decode's experts as mailboxes (mailboxes.py): with every expert on
    the CPU, bit-identical to the plain engine (each reply is the fused
    kernel's own term, summed in its order). On a GPU also with half of
    every layer's experts in GPU pools, against the plain GPU engine: the
    rounding class (a GPU's replies may differ in the last bf16 bit)."""
    import torch
    keep = []
    got = _with_env({"VXPU_EXPERT_MAILBOXES": "1", "VXPU_GPU_EXPERTS": None}, record_vxpu_engine,
                    model, "main", prompts, settings, artifact, keep=keep)
    selected = {s["impl"] for s in keep[0].plan.explain(keep[0].plan_ops())["ops"]["vxpu::moe_experts"]["selected"]}
    if "mailboxes" not in selected:
        print(f"expert mailboxes check: the mailboxes implementation was not selected ({sorted(selected)})")
        return False
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]])
    rows, _ = compare(plain, got, strict_tokens=True)
    print("expert mailboxes, every expert on the CPU, against the plain engine (must be bit-identical):")
    print(table(plain, got, rows))
    if not same:
        print("expert mailboxes check: not bit-identical with every expert on the CPU")
        return False
    if not torch.cuda.is_available():
        return True
    with open(os.path.join(artifact, "binding.json")) as f:
        experts = (json.load(f).get("external") or {}).get("experts") or {}
    n = len(next(iter(next(iter(experts.values())).values()))["parts"])
    gpus = min(torch.cuda.device_count(), 2)
    held = list(range(0, n, 2))
    path = os.path.join(workdir, "placement-mailboxes.json")
    with open(path, "w") as f:
        json.dump({"format": "vxpu-placement/2",
                   "gpus": [{"layers": {layer: held[g::gpus] for layer in experts}} for g in range(gpus)]}, f)
    ref = _with_env({"VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}, record_vxpu_engine,
                    model, "main", prompts, settings, artifact, device="cuda")
    pooled = _with_env({"VXPU_EXPERT_MAILBOXES": "1", "VXPU_GPU_EXPERTS": path}, record_vxpu_engine,
                       model, "main", prompts, settings, artifact, device="cuda")
    rows, ok = compare(ref, pooled, strict_tokens=True)
    print(f"expert mailboxes with half the experts in {gpus} GPU pool(s), against the plain GPU engine:")
    print(table(ref, pooled, rows))
    return ok


def _check_expert_remote(model, prompts, settings, artifact, plain):
    """Experts answered by another process (expert_server.py): a server
    holding every other layer's banks, on a local port, and the engine,
    which does not load those layers' banks, sending it every call of
    theirs (prefill included) through its mailboxes. The server computes
    with the same CPU kernels, so tokens and logits must be bit-identical
    to the plain engine's."""
    import subprocess
    import sys
    import threading
    with open(os.path.join(artifact, "binding.json")) as f:
        layers = sorted(int(x) for x in ((json.load(f).get("external") or {}).get("experts") or {}))
    if not layers:
        return True
    served = layers[::2]
    proc = subprocess.Popen([sys.executable, "-m", "vxpu.expert_server", artifact, "--host", "127.0.0.1",
                             "--port", "0", "--layers", ",".join(map(str, served))],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, started = [], threading.Event()

    def read():  # the server's output, kept for a failure; the address once it listens
        for line in proc.stdout:
            lines.append(line)
            if "expert server on " in line:
                started.set()
        started.set()  # it exited

    threading.Thread(target=read, daemon=True).start()
    try:
        started.wait(timeout=300)
        address = next((ln.split("expert server on ", 1)[1].split()[0].rstrip(":") for ln in lines
                        if "expert server on " in ln), None)
        if address is None or ":" not in address:
            print("expert remote check: the expert server did not start:\n" + "".join(lines[-20:]))
            return False
        keep = []
        got = _with_env({"VXPU_EXPERT_MAILBOXES": "1", "VXPU_EXPERT_REMOTE": address, "VXPU_GPU_EXPERTS": None},
                        record_vxpu_engine, model, "main", prompts, settings, artifact, keep=keep)
        loaded = [layer for layer in served if not keep[0].external.experts[layer].w1.is_meta]
        if loaded:
            print(f"expert remote check: layers {loaded} are on the expert server but were loaded here too")
            return False
        if not _check_expert_server_errors(address, artifact, served[0]):
            return False
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]])
    rows, _ = compare(plain, got, strict_tokens=True)
    print(f"experts of layers {served} answered by an expert server at {address}, against the plain engine "
          "(must be bit-identical):")
    print(table(plain, got, rows))
    if not same:
        print("expert remote check: not bit-identical")
    return same


def _check_expert_server_errors(address, artifact, layer):
    """The expert server refuses a request it cannot serve with a reason
    the engine hears (a layer it does not hold), then closes; the next
    request is answered on a new connection."""
    import torch
    from . import expert_server, mailboxes
    with open(os.path.join(artifact, "manifest.json")) as f:
        owner = mailboxes.RemoteOwner(address, checkpoint=expert_server.checkpoint_id(json.load(f)))
    experts, _ = expert_server.load(artifact, {layer})
    moe = experts[layer]
    x = torch.zeros(1, moe.dim, dtype=torch.bfloat16)
    one = torch.zeros(1, dtype=torch.long)
    try:
        owner.answer(moe, x, one, one + moe.experts_start_idx, torch.ones(1), layer=10_000)
    except RuntimeError as e:
        if "not held here" not in str(e):
            print(f"expert remote check: unexpected refusal: {e}")
            return False
    else:
        print("expert remote check: a request for a layer the server does not hold was answered")
        return False
    if owner.answer(moe, x, one, one + moe.experts_start_idx, torch.ones(1), layer=layer).shape != (1, moe.dim):
        print("expert remote check: no answer after the refused request")
        return False
    owner.close()
    return True


def _check_step_graph(model, prompts, settings, artifact, workdir):
    """On a GPU: the whole decode step in one CUDA graph (stepgraph.py),
    the AOTInductor segments and the host ops as mailboxes the GPU waits
    on, against the same dataflow over the same segments run node by
    node: tokens and logits bit-identical, and the step really captured.
    Skipped without CUDA."""
    import torch
    if not torch.cuda.is_available():
        print("step graph check skipped: no CUDA")
        return True
    base = {"VXPU_DATAFLOW": "1", "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "aoti"),
            "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}
    ref = _with_env({**base, "VXPU_STEP_GRAPH": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True)
    keep = []
    got = _with_env({**base, "VXPU_STEP_GRAPH": "1"}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True, keep=keep)
    if not isinstance(getattr(keep[0]._decode_split, "graph", None), torch.cuda.CUDAGraph):
        print("step graph check: the decode step was never captured")
        return False
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in ref["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in ref["prompts"]])
    rows, _ = compare(ref, got, strict_tokens=True)
    print("the decode step as one CUDA graph, against the same segments run node by node (must be bit-identical):")
    print(table(ref, got, rows))
    import gc
    worker = keep[0]._decode_split._worker
    keep.clear()  # the engine dropped: its step graph's host thread must let go of the model
    gc.collect()
    worker.join(timeout=10)
    if worker.is_alive():
        print("step graph check: the host thread outlived its engine")
        return False
    return same


def _check_step_graph_pools(model, prompts, settings, artifact, workdir):
    """On a GPU: the step graph with half of every layer's experts in GPU
    pools (split between two GPUs where there are two: the other GPU's
    graph takes part), the CPU answering the other pairs through its
    mailbox, against the same dataflow run node by node with expert
    mailboxes and the same placement (the same per-pair replies, summed
    in the same order): bit-identical."""
    import torch
    if not torch.cuda.is_available():
        return True
    with open(os.path.join(artifact, "binding.json")) as f:
        experts = (json.load(f).get("external") or {}).get("experts") or {}
    if not experts:
        return True
    n = len(next(iter(next(iter(experts.values())).values()))["parts"])
    gpus = min(torch.cuda.device_count(), 2)  # with two, half on each: the other GPU's graph takes part too
    held = list(range(0, n, 2))
    path = os.path.join(workdir, "placement-step-graph.json")
    with open(path, "w") as f:
        json.dump({"format": "vxpu-placement/2",
                   "gpus": [{"layers": {layer: held[g::gpus] for layer in experts}} for g in range(gpus)]}, f)
    base = {"VXPU_DATAFLOW": "1", "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "aoti"),
            "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": "1", "VXPU_GPU_EXPERTS": path}
    ref = _with_env({**base, "VXPU_STEP_GRAPH": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True, telemetry=True)
    keep = []
    got = _with_env({**base, "VXPU_STEP_GRAPH": "1"}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True, telemetry=True, keep=keep)
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in ref["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in ref["prompts"]])
    rows, _ = compare(ref, got, strict_tokens=True)
    print(f"the step graph with half the experts in GPU pools on {gpus} GPU(s), against the same run node by node "
          "(must be bit-identical):")
    print(table(ref, got, rows))
    # the experts nodes answered in the graph record their selections too:
    # the statistics placements are made from see the same as node by node
    selections = [[(p.get("telemetry") or {}).get("experts", {}).get("decode") for p in d["prompts"]] for d in (ref, got)]
    if selections[0] != selections[1] or not any(selections[0]):
        print("step graph pools check: decode expert selections differ from the node-by-node run's")
        return False
    return same and isinstance(getattr(keep[0]._decode_split, "graph", None), torch.cuda.CUDAGraph)


def _check_dataflow(model, prompts, settings, artifact, plain):
    """The engine's decode run as a dataflow graph (dataflow.py): the same
    tokens and logits as the plain run, every edge through an inbox. On a
    GPU also with the host-op nodes placed on the CPU: their inputs and
    outputs then cross devices through the inboxes (the GPU run first
    without, as the reference)."""
    import torch
    keep = []
    got = _with_env({"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": None}, record_vxpu_engine,
                    model, "main", prompts, settings, artifact, keep=keep)
    runner = keep[0]._decode_split
    rows, ok = compare(plain, got, strict_tokens=True)
    plan = runner.inbox_plan()
    print(f"{runner.graph.describe()}, {len(plan)} inboxes, {sum(i['bytes'] for i in plan)} bytes a step, "
          "against the plain engine:")
    print(table(plain, got, rows))
    if not torch.cuda.is_available() or not any(n.kind != "dense" for n in runner.graph.nodes):
        return ok
    ref = _with_env({"VXPU_DATAFLOW": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda")
    keep = []
    placed = _with_env({"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": "experts=cpu,engram=cpu"}, record_vxpu_engine,
                       model, "main", prompts, settings, artifact, device="cuda", keep=keep)
    crossing = sum(i["bytes"] for i in keep[0]._decode_split.inbox_plan() if not i["device"].startswith("cuda"))
    rows2, ok2 = compare(ref, placed, strict_tokens=True)
    print(f"dataflow on the GPU with the host-op nodes on the CPU ({crossing} bytes a step into CPU inboxes), "
          "against the plain GPU engine:")
    print(table(ref, placed, rows2))
    return ok and ok2


def _with_env(env, fn, *args, **kwargs):
    """fn(*args, **kwargs) with env set (None: unset), restored after."""
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return fn(*args, **kwargs)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _check_mover_lifetime():
    """The expert mover (experts_gpu.Mover) is freed with its engine: its
    worker thread holds it only weakly, so the pools and banks it reaches
    go with it, and the thread exits."""
    import gc
    import threading
    import weakref
    from . import experts_gpu
    mover = experts_gpu.Mover(2, 0.25)
    ref = weakref.ref(mover)
    del mover
    gc.collect()
    if ref() is not None:
        raise SystemExit("expert mover: still referenced after its engine let it go (the worker holds it)")
    deadline = time.time() + 5
    while any(t.name == "vxpu-expert-moves" for t in threading.enumerate()) and time.time() < deadline:
        time.sleep(0.1)
    if any(t.name == "vxpu-expert-moves" for t in threading.enumerate()):
        raise SystemExit("expert mover: its worker thread outlived it")


def _check_hardware():
    import json
    from .hardware import CATALOG, probe, _match_catalog, Machine, CPU

    for name, m in CATALOG.items():
        m.validate()
        s = m.summary()
        if not s.startswith(f"Machine: {m.name}"):
            raise SystemExit(f"{name}: summary did not start with Machine name")
        decoded = json.loads(m.to_json())
        if decoded["name"] != m.name:
            raise SystemExit(f"{name}: JSON round-trip mismatch: {decoded['name']!r} != {m.name!r}")

    current = probe()
    current.validate()
    current.summary()
    if not current.name.startswith("probed:"):
        raise SystemExit(f"expected probed machine name to start with 'probed:', got {current.name!r}")
    if current.measured is not None:
        raise SystemExit("probed machine must not inherit catalog benchmark measurements")

    for cpu, threads, want in [
        ("AMD EPYC 9B14 64-Core Processor", 60, "c3d-standard-60"),
        ("AMD EPYC 9B14 64-Core Processor", 180, "c3d-standard-180"),
        ("AMD EPYC 9B45 48-Core Processor", 48, "g4-standard-48"),
        ("AMD EPYC 9B45 48-Core Processor", 96, "g4-standard-96"),
        ("Apple M4 Pro", 14, "mac-m4-pro"),
        ("AMD EPYC 7B13 64-Core Processor", 32, None),  # Milan: not in catalog
        ("AMD Ryzen 9 7950X 16-Core Processor", 32, None),
        ("Apple M1", 8, None),
        ("Intel(R) Xeon(R) Platinum 8281 CPU @ 2.70GHz", 32, None),
    ]:
        got = _match_catalog({"cpu": cpu, "cpu_threads": threads})
        got_name = got.name if got else None
        if got_name != want:
            raise SystemExit(f"{cpu!r} x{threads}: matched {got_name!r}, want {want!r}")

    for gpu, count, want in [
        ("NVIDIA RTX PRO 6000", 2, "g4-standard-96"),
        ("NVIDIA RTX PRO 6000", 1, "g4-standard-48"),
        ("NVIDIA H100 80GB HBM3", 8, "a3-highgpu-8g"),
        ("NVIDIA H100 80GB HBM3", 1, None),
        ("NVIDIA A100-SXM4-40GB", 1, "a2-highgpu-1g"),
        ("NVIDIA A100-SXM4-40GB", 8, None),
        ("NVIDIA L4", 1, None),
        ("Tesla T4", 1, None),
        ("NVIDIA RTX A6000", 1, None),
    ]:
        got = _match_catalog({"gpu": gpu, "gpu_count": count})
        got_name = got.name if got else None
        if got_name != want:
            raise SystemExit(f"{gpu!r} x{count}: matched {got_name!r}, want {want!r}")

    try:
        Machine(name="bad", cpu=CPU("cpu", 0, 0, "sse"), ram_bytes=-1)
        raise SystemExit("expected Machine validation failure on bad parameters")
    except ValueError:
        pass

    test_m = Machine(
        name="test-summary",
        cpu=CPU("test-cpu", 4, 4, "sse", 50e9, {"l1": 1024}),
        ram_bytes=16 * 1024**3,
        measured={"comms_path": "gcs", "items": 12, "time_ms": 5.2, "bytes_per_sec": 1e9},
    )
    s = test_m.summary()
    if "comms_path: gcs" not in s or "items: 12" not in s:
        raise SystemExit("summary did not format non-numeric measured fields properly")


def selftest(workdir, settings):
    """The CI tier: tiny models through export, the in-process engine,
    and a transformers reference, token-exact. No cluster."""
    _check_compare()
    _check_mover_lifetime()
    _check_compile_plan()
    _check_hardware()
    _check_compile_names()
    os.makedirs(workdir, exist_ok=True)
    _check_host_reads(workdir)
    _check_local_blobs_copy(workdir)
    _check_expert_sim()
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
    try:
        # DeepSeek-V4.1 layout through the model maker's module (third_party fork):
        # fp8 dense, fp4 experts, compressed sparse attention, engram table
        from .fixtures import tiny_deepseek_v41
        models.append(("tiny-deepseek-v41", tiny_deepseek_v41.write(os.path.join(workdir, "tiny-deepseek-v41")), "ids"))
    except ImportError as e:
        print(f"tiny-deepseek-v41 fixture skipped: {e}")
    failed = []
    for name, model, prompts in models:
        artifact = os.path.join(workdir, name + "-artifact")
        export_artifact(model, artifact, max_cache_len=1024)
        if name == "tiny-deepseek-v41":  # no transformers model: the fork run eagerly is the reference
            ref = record_deepseek_eager(model, "main", prompts, settings)
        else:
            ref = record_transformers(model, "main", prompts, settings)
        got = record_vxpu_engine(model, "main", prompts, settings, artifact)
        for label, doc in (("reference", ref), ("candidate", got)):
            with open(os.path.join(workdir, f"{name}-{label}.json"), "w") as f:
                json.dump(doc, f, indent=1)
        with open(os.path.join(artifact, "binding.json")) as f:
            if json.load(f).get("prefill_from_zero"):
                _check_continuation(artifact, ID_PROMPTS[0]["ids"])
        rows, ok = compare(ref, got, strict_tokens=True)
        print(table(ref, got, rows))
        if not ok:
            failed.append(name)
        if name == "tiny-fp8-moe" and not _check_local_blobs(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (local weight tier)")
        if not _check_dataflow(model, prompts, settings, artifact, got):
            failed.append(name + " (dataflow)")
        if name == "tiny-deepseek-v41" and not _check_expert_mailboxes(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (expert mailboxes)")
        if name == "tiny-deepseek-v41" and not _check_expert_remote(model, prompts, settings, artifact, got):
            failed.append(name + " (expert remote)")
        if name == "tiny-deepseek-v41" and not _check_step_graph(model, prompts, settings, artifact, workdir):
            failed.append(name + " (step graph)")
        if name == "tiny-deepseek-v41" and not _check_step_graph_pools(model, prompts, settings, artifact, workdir):
            failed.append(name + " (step graph with pools)")
        if name == "tiny-llama" and not _check_chat_telemetry(artifact):
            failed.append(name + " (chat telemetry)")
        if name == "tiny-deepseek-v41" and not _check_gpu_experts(model, prompts, settings, artifact, workdir):
            failed.append(name + " (gpu experts)")
        if name == "tiny-deepseek-v41" and not _check_telemetry(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (telemetry)")
        if name == "tiny-deepseek-v41" and not _check_lowering(model, prompts, settings, artifact, workdir):
            failed.append(name + " (lowered)")
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
    rec.add_argument("--producer", choices=["transformers", "vxpu-engine", "vxpu-router", "vllm", "deepseek-eager"], required=True)
    rec.add_argument("--model", required=True, help="Hub repo id or local checkpoint directory")
    rec.add_argument("--revision", default="main")
    rec.add_argument("--prompts", choices=["auto", "text", "ids"], default="auto",
                     help="text prompts through the chat template, or fixed ids (auto: ids if no tokenizer)")
    rec.add_argument("--artifact", help="vxpu artifact directory (vxpu-* producers)")
    rec.add_argument("--router", help="router host:port (vxpu-router; default: port-forward)")
    rec.add_argument("--device", default="cpu", help="transformers / vxpu-engine producers (vllm places itself)")
    rec.add_argument("--dtype", default="auto", help="transformers / vllm model dtype (auto: the config's)")
    rec.add_argument("--compile", action="store_true", help="vxpu-engine: torch.compile the decode graph")
    rec.add_argument("--telemetry", action="store_true",
                     help="vxpu-engine / vxpu-router: keep each prompt's execution telemetry (vxpu-telemetry/1)")
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
