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

"""Export a model to a vXPU artifact — without loading its weights.

The model is instantiated on torch's meta device (shapes and dtypes,
no storage), so a laptop can export any size of model. Two graphs are
captured via transformers' export wrapper:

  prefill.pt2 — dynamic sequence length, explicit cache positions
  decode.pt2  — constant shapes, so executors can compile it once

Every tensor the graphs reference is classified:

  bound   — a content-addressed weight reference from the manifest
  derived — computed from config at load time (RoPE tables, embedding
            scales); never stored in checkpoints
  state   — per-session, zero-initialized (KV caches, counters);
            mutated in place by the graphs

An artifact directory is the unit of interchange:
manifest.json + binding.json + prefill.pt2 + decode.pt2 — a few tens
of MB describing models of any size.

Usage:
    python -m vxpu.export google/gemma-4-E4B-it -o gemma-e4b/
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import executorch as hf_executorch
from transformers.integrations.executorch import (
    TorchExportableModuleForDecoderOnlyLM,
)

from .fp8 import (apply_fp8, fp8_config, scale_dtype_of, set_experts_mode,
                  stacked_sources)
from .manifest import build_manifest

# Buffers computed from config at init time, never stored in
# checkpoints. Executors recompute them (see rehydrate.derived_tensor).
DERIVED_MARKERS = ("rotary_emb", "inv_freq", "embed_scale", "softcap",
                   "inv_timescales")


def _head_shapes(config):
    """Per-layer (num_kv_heads, head_dim) for the static cache.

    transformers' own helper slices ``layer_types[:-num_kv_shared_layers]``,
    which is empty when a model shares no KV layers (``[:-0]`` is ``[:0]``)
    — e.g. Gemma 4 31B. Fall back to the full layer list in that case;
    otherwise defer to transformers.
    """
    num_heads, head_dim = _upstream_head_shapes(config)
    if not isinstance(num_heads, list) or num_heads:
        return num_heads, head_dim
    layer_types = list(config.layer_types)
    shared = getattr(config, "num_kv_shared_layers", 0) or 0
    if shared:
        layer_types = layer_types[:-shared]
    head_dim = [config.global_head_dim
                if layer == "full_attention" and config.global_head_dim
                else config.head_dim for layer in layer_types]
    num_heads = [config.num_global_key_value_heads
                 if layer == "full_attention" and config.attention_k_eq_v
                 else config.num_key_value_heads for layer in layer_types]
    return num_heads, head_dim


_upstream_head_shapes = hf_executorch.get_head_shapes
hf_executorch.get_head_shapes = _head_shapes


def classify(exported, manifest, num_experts=0):
    """Classify every tensor the exported program references.

    Returns (bound, derived, state, stacked, unbound). ``stacked`` maps a
    stacked expert parameter to the checkpoint tensors it is assembled
    from (see fp8.stacked_sources).
    """
    sig = exported.graph_signature
    # Tied weights collapse to one placeholder in the signature but the
    # state dict keeps both names — classify over the union.
    param_fqns = set(sig.inputs_to_parameters.values()) | set(
        exported.state_dict.keys())
    fqns = list(dict.fromkeys(
        list(param_fqns)
        + list(sig.inputs_to_buffers.values())
        + list(exported.constants.keys())))

    def lookup(fqn):
        # Wrapper nesting adds prefixes; match by peeling components.
        parts = fqn.split(".")
        for i in range(len(parts)):
            ref = manifest["tensors"].get(".".join(parts[i:]))
            if ref is not None:
                return ref
        return None

    tied_ref = next(
        (ref for name, ref in manifest["tensors"].items()
         if name.endswith("embed_tokens.weight")), None)

    bound, derived, state, stacked, unbound = {}, [], [], {}, []
    for fqn in fqns:
        ref = lookup(fqn)
        if ref is None and fqn.endswith("lm_head.weight"):
            ref = tied_ref  # weight tying: an alias, not a second copy
        parts = (stacked_sources(fqn, manifest["tensors"], num_experts)
                 if ref is None and num_experts else None)
        if ref is not None:
            bound[fqn] = ref
        elif parts is not None:
            stacked[fqn] = {"parts": parts, "cat_dim": 0}
        elif any(marker in fqn for marker in DERIVED_MARKERS):
            derived.append(fqn)
        elif fqn not in param_fqns:
            state.append(fqn)
        else:
            unbound.append(fqn)
    return bound, derived, state, stacked, unbound


def export_artifact(repo_id, out_dir, max_cache_len=1024, revision="main"):
    """Build the complete artifact directory for repo_id."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = build_manifest(repo_id, revision)
    config = AutoConfig.from_pretrained(repo_id, revision=revision)

    quantization = fp8_config(config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config).eval()
        # A model-maker fp8 checkpoint: the graph consumes its fp8
        # weights and block scales directly (see fp8.py).
        swapped = (apply_fp8(model, config, scale_dtype_of(manifest["tensors"]))
                   if quantization else 0)
        model.generation_config.cache_implementation = "static"
        exportable = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=max_cache_len)

        seq = torch.export.Dim("seq", min=1, max=max_cache_len)
        set_experts_mode(model, "prefill")
        prefill = exportable.export(
            input_ids=torch.zeros(1, 8, dtype=torch.long),
            cache_position=torch.arange(8),
            dynamic_shapes={"input_ids": {1: seq},
                            "cache_position": {0: seq}})
        set_experts_mode(model, "decode")
        decode = exportable.export(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            cache_position=torch.zeros(1, dtype=torch.long))

    torch.export.save(prefill, os.path.join(out_dir, "prefill.pt2"))
    torch.export.save(decode, os.path.join(out_dir, "decode.pt2"))

    text_config = (config.get_text_config()
                   if hasattr(config, "get_text_config") else config)
    num_experts = getattr(text_config, "num_experts", 0) or 0
    bound, derived, state, stacked, unbound = classify(
        decode, manifest, num_experts)
    if unbound:
        raise RuntimeError(
            f"unbound tensors (artifact would be incomplete): {unbound}")
    binding = {"bound": bound, "derived": derived, "state": state,
               "stacked": stacked, "max_cache_len": max_cache_len}
    if quantization:
        binding["quantization"] = {
            "quant_method": "fp8",
            "block_size": list(quantization["block_size"]),
            "fp8_modules": swapped}

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "binding.json"), "w") as f:
        json.dump(binding, f, indent=1, sort_keys=True)
    return manifest, binding


def main():
    parser = argparse.ArgumentParser(
        description="Export a model to a vXPU artifact (no weights "
                    "downloaded)")
    parser.add_argument("repo_id")
    parser.add_argument("-o", "--out", required=True,
                        help="artifact output directory")
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    manifest, binding = export_artifact(
        args.repo_id, args.out, args.max_cache_len, args.revision)
    total = sum(f["size"] for f in manifest["files"].values())
    print(f"artifact: {args.out}")
    print(f"  model: {args.repo_id} ({len(manifest['tensors'])} tensors, "
          f"{total / 1e9:.2f} GB of weights referenced, not included)")
    print(f"  binding: {len(binding['bound'])} bound, "
          f"{len(binding['derived'])} derived, "
          f"{len(binding['state'])} state, "
          f"{len(binding['stacked'])} stacked"
          + (f" (fp8, {binding['quantization']['fp8_modules']} modules)"
             if binding.get("quantization") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
