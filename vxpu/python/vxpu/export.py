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

from .fp8 import apply_fp8, fp8_config, set_experts_implementation
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


def checkpoint_view(model, manifest_tensors):
    """The checkpoint as the model's parameters see it, per transformers.

    ``from_pretrained`` does not look checkpoint keys up by name: it runs
    the model's conversion mapping, which renames keys (``model.
    language_model.X`` -> ``model.X`` for a text-only class) and merges
    several checkpoint tensors into one parameter (per-expert
    projections into an expert bank, a sharded table into one table).
    Replay exactly that on the manifest, without the bytes.

    Returns (direct, merged): ``direct`` maps a model key to the manifest
    name holding it; ``merged`` maps a model key to an assembly spec,
    ``{"parts": [[names...], ...], "cat_dim": d, "stack": bool}`` (rows
    concatenated along ``cat_dim``, then stacked along a new leading
    dimension if ``stack``), as the executor's fetch_stacked expects.
    """
    import re
    from transformers.conversion_mapping import get_model_conversion_mapping

    renamings = get_model_conversion_mapping(model)
    direct, groups = {}, {}
    for name in sorted(manifest_tensors):
        key, converter, pattern = name, None, None
        for renaming in renamings:
            new_key, matched = renaming.rename_source_key(key)
            if matched is not None and getattr(renaming, "operations", None):
                converter, pattern = renaming, matched
            key = new_key
        if converter is None:
            direct[key] = name
            continue
        group = groups.setdefault(key, {"converter": converter, "sources": {}})
        # The index the pattern's "*" stood for, so shards and experts
        # assemble in numeric order (shard_10 after shard_9).
        regex = re.escape(pattern).replace(r"\*", r"([^.]+)")
        match = re.search(regex, name)
        index = match.group(1) if match else "0"
        index = int(index) if index.isdigit() else index
        group["sources"].setdefault(pattern, []).append((index, name))

    merged = {}
    for key, group in groups.items():
        ops = [type(op).__name__ for op in group["converter"].operations]
        columns = [[n for _, n in sorted(v)] for v in group["sources"].values()]
        if ops == ["MergeModulelist", "Concatenate"]:
            # Stack each pattern's tensors over the index, then concatenate the
            # patterns along `dim` of the stacked result: per row that is a
            # concatenation along dim - 1.
            dim = group["converter"].operations[1].dim
            rows = [list(parts) for parts in zip(*columns)]
            merged[key] = {"parts": rows, "cat_dim": dim - 1, "stack": True}
        elif ops == ["MergeModulelist"]:
            merged[key] = {"parts": [[n] for n in columns[0]], "cat_dim": 0,
                           "stack": True}
        elif ops == ["Concatenate"]:
            dim = group["converter"].operations[0].dim
            merged[key] = {"parts": [[n for col in columns for n in col]],
                           "cat_dim": dim, "stack": False}
        else:
            raise NotImplementedError(
                f"checkpoint conversion {ops} for {key} is not supported")
    return direct, merged


def constant_aliases(exportable, *programs):
    """Lifted constants that are the same tensor as a wrapper buffer.

    Non-strict export captures a tensor reached through a Python object
    (the static cache's layers) as a lifted constant, even when the same
    tensor is also a registered buffer of the wrapper; both names end up
    in the graph. They are one tensor and must stay one on the executor
    (a cache written under one name and read under the other would not
    be a cache), so map each alias to the buffer it is.
    """
    buffers = {id(t): name for name, t in exportable.model.named_buffers()}
    aliases = {}
    for program in programs:
        for name, tensor in program.constants.items():
            canonical = buffers.get(id(tensor))
            if canonical is not None and canonical != name:
                if aliases.get(name, canonical) != canonical:
                    raise RuntimeError(
                        f"constant {name} aliases different buffers in the "
                        f"two graphs ({aliases[name]} vs {canonical})")
                aliases[name] = canonical
    return aliases


MIN_PREFILL_LEN = 2


def classify(exported, manifest, view, skip=()):
    """Classify every tensor the exported program references.

    Returns (bound, derived, state, stacked, unbound). ``view`` is the
    checkpoint as the model's parameters see it (checkpoint_view);
    ``stacked`` maps a merged parameter to its assembly spec. ``skip``
    names are aliases of other tensors and are not classified.
    """
    direct, merged = view
    sig = exported.graph_signature
    # Tied weights collapse to one placeholder in the signature but the
    # state dict keeps both names — classify over the union.
    param_fqns = set(sig.inputs_to_parameters.values()) | set(
        exported.state_dict.keys())
    fqns = list(dict.fromkeys(
        list(param_fqns)
        + list(sig.inputs_to_buffers.values())
        + list(exported.constants.keys())))

    def resolve(fqn, table):
        # Wrapper nesting adds prefixes; match by peeling components.
        parts = fqn.split(".")
        for i in range(len(parts)):
            value = table.get(".".join(parts[i:]))
            if value is not None:
                return value
        return None

    def lookup(fqn):
        name = resolve(fqn, direct)
        return manifest["tensors"][name] if name is not None else None

    tied_ref = next(
        (ref for name, ref in manifest["tensors"].items()
         if name.endswith("embed_tokens.weight")), None)

    bound, derived, state, stacked, unbound = {}, [], [], {}, []
    for fqn in fqns:
        if fqn in skip:
            continue
        ref = lookup(fqn)
        if ref is None and fqn.endswith("lm_head.weight"):
            ref = tied_ref  # weight tying: an alias, not a second copy
        spec = resolve(fqn, merged) if ref is None else None
        if ref is not None:
            bound[fqn] = ref
        elif spec is not None:
            stacked[fqn] = spec
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
    # Size-oblivious reasoning for the prefill's symbolic length: without
    # it export case-splits "could this dimension be 1?" on derived sizes
    # (a chunk count of a linear-attention scan) and specializes the
    # length to the example's range.
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True
    # The model's compute dtype. Composite configs (text + vision) may
    # carry it only on the text config; from_pretrained would read the
    # checkpoint's dtypes, which a weightless export cannot.
    text_config = (config.get_text_config()
                   if hasattr(config, "get_text_config") else config)
    dtype = (getattr(config, "dtype", None)
             or getattr(text_config, "dtype", None) or torch.bfloat16)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype).eval()
        # A model-maker fp8 checkpoint: transformers' own fp8 modules,
        # whose graph consumes the fp8 weights and block scales directly
        # (see fp8.py and third_party/transformers).
        view = checkpoint_view(model, manifest["tensors"])
        swapped = (apply_fp8(model, config, view, manifest["tensors"])
                   if quantization else 0)
        model.generation_config.cache_implementation = "static"
        exportable = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=max_cache_len)

        # A prefill of one token is served by the decode graph (see the
        # engine), so the prefill graph starts at two: the symbolic
        # reasoner otherwise case-splits "is the length 1?" at every
        # transposed-then-reshaped activation.
        seq = torch.export.Dim("seq", min=MIN_PREFILL_LEN, max=max_cache_len)
        # Experts: grouped_mm for prefill (routed pairs sorted by expert,
        # one grouped matmul per projection), batched_mm for decode (one
        # gathered block per pair). Both are static-shape; both are the
        # sparse routing transformers runs eagerly.
        set_experts_implementation(model, "grouped_mm")
        # Non-strict export runs the model's Python as is on fake tensors
        # (strict mode re-traces it through Dynamo, which trips on the
        # Python-heavy hybrid models); the captured graph is the same.
        prefill = exportable.export(
            input_ids=torch.zeros(1, 8, dtype=torch.long),
            cache_position=torch.arange(8),
            dynamic_shapes={"input_ids": {1: seq},
                            "cache_position": {0: seq}},
            strict=False)
        set_experts_implementation(model, "batched_mm")
        # Linear-attention / conv states branch in Python on whether a
        # previous state exists; the decode graph is the "yes" branch.
        set_state_flag = getattr(exportable.model, "set_has_previous_state", None)
        if set_state_flag is not None:
            set_state_flag(True)
        decode = exportable.export(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            cache_position=torch.zeros(1, dtype=torch.long),
            strict=False)

    torch.export.save(prefill, os.path.join(out_dir, "prefill.pt2"))
    torch.export.save(decode, os.path.join(out_dir, "decode.pt2"))
    aliases = constant_aliases(exportable, prefill, decode)

    bound, derived, state, stacked, unbound = classify(
        decode, manifest, view, skip=set(aliases))
    if unbound:
        raise RuntimeError(
            f"unbound tensors (artifact would be incomplete): {unbound}")
    binding = {"bound": bound, "derived": derived, "state": state,
               "stacked": stacked, "aliases": aliases,
               "max_cache_len": max_cache_len,
               "min_prefill_len": MIN_PREFILL_LEN}
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
          f"{len(binding['stacked'])} stacked, "
          f"{len(binding['aliases'])} aliases"
          + (f" (fp8, {binding['quantization']['fp8_modules']} modules)"
             if binding.get("quantization") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
