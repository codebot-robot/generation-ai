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

"""Model-maker fp8 checkpoints (``quant_method: fp8``) in the exported graph.

transformers already has the modules for these checkpoints (``FP8Linear``,
``FP8Experts`` in ``integrations/finegrained_fp8.py``), and its loader
installs them in ``from_pretrained``. Two things stood between them and a
weightless export on a GPU-less machine, and both are fixed in the
transformers fork under ``third_party/transformers`` rather than here:

* their forward dispatched only to CUDA kernels (DeepGEMM, a Triton
  kernel from the Hub), so off the GPU there was no computation to
  trace; the fork adds a plain-aten reference path (dequantize the
  block, matmul) used on CPU, MPS and the meta device;
* the grouped experts counted tokens with an integer ``histc`` off the
  CPU, which has no CPU kernel; the fork counts in float off the GPU.

What remains in vxpu is bookkeeping: install the modules on the meta
model the way the loader would (``apply_fp8``) and choose the experts
implementation per graph (``grouped_mm`` for prefill, ``batched_mm`` for
decode: both static-shape and the same routing computation transformers
runs). How a merged parameter (an expert bank, a sharded table) is
assembled from checkpoint tensors comes from transformers' own
conversion mapping (export.checkpoint_view).
"""



def fp8_config(config):
    """The checkpoint's fp8 settings, or None if it is not an fp8 checkpoint."""
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return None
    qc = qc if isinstance(qc, dict) else qc.to_dict()
    if qc.get("quant_method") != "fp8":
        return None
    return {"block_size": tuple(qc.get("weight_block_size") or (128, 128)),
            "modules_to_not_convert": set(qc.get("modules_to_not_convert") or [])}


def _resolve(fqn, table):
    """Find a model key in ``table`` for a module path, peeling wrapper prefixes."""
    parts = fqn.split(".")
    for i in range(len(parts)):
        value = table.get(".".join(parts[i:]))
        if value is not None:
            return value
    return None


def apply_fp8(model, config, view, manifest_tensors):
    """Install transformers' fp8 modules on a (meta) model, as its loader does.

    Runs the same HfQuantizer preprocessing ``from_pretrained`` runs. The
    loader additionally lets the *checkpoint* decide: a linear whose
    weight has no ``weight_scale_inv`` beside it (Qwen's bf16 lm_head,
    with an empty ``modules_to_not_convert``) stays a plain linear, and
    an embedding table stored as fp8 with a per-tensor ``weight_scale``
    (Qwen3.8's n-gram table) becomes an ``FP8Embedding``. ``view`` is the
    checkpoint as the model's parameters see it (export.checkpoint_view).
    Returns the number of modules replaced (0 for a non-fp8 checkpoint).
    """
    import torch.nn as nn
    from transformers import FineGrainedFP8Config
    from transformers.integrations import finegrained_fp8
    from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear
    from transformers.quantizers import AutoHfQuantizer

    settings = fp8_config(config)
    if settings is None:
        return 0
    if not hasattr(finegrained_fp8, "_reference_fp8_linear"):
        raise RuntimeError(
            "this transformers has no off-GPU fp8 path, so an fp8 checkpoint "
            "cannot be exported on the meta device; install the fork: "
            "pip install -e third_party/transformers (see docs/representation.md)")
    direct, merged = view

    def has(key):
        return _resolve(key, direct) is not None or _resolve(key, merged) is not None

    def dtype_of(key):
        name = _resolve(key, direct)
        if name is None:
            spec = _resolve(key, merged)
            name = spec["parts"][0][0] if spec else None
        return manifest_tensors[name]["dtype"] if name else None

    qc = config.quantization_config
    qc = FineGrainedFP8Config.from_dict(qc) if isinstance(qc, dict) else qc
    skip = list(qc.modules_to_not_convert or [])
    embeddings = []
    for fqn, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            probe = fqn
        elif isinstance(getattr(mod, "gate_up_proj", None), nn.Parameter):
            probe = fqn  # merged bank: experts.gate_up_proj[_scale_inv]
            probe_name = f"{fqn}.gate_up_proj"
            if has(probe_name) and not has(probe_name + "_scale_inv"):
                skip.append(fqn)
            continue
        elif isinstance(mod, nn.Embedding):
            if (dtype_of(f"{fqn}.weight") == "F8_E4M3"
                    and has(f"{fqn}.weight_scale")):
                embeddings.append((fqn, mod))
            continue
        else:
            continue
        if has(probe + ".weight") and not has(probe + ".weight_scale_inv"):
            skip.append(fqn)
    qc.modules_to_not_convert = sorted(set(skip))
    before = sum(isinstance(m, (FP8Linear, FP8Experts)) for m in model.modules())
    quantizer = AutoHfQuantizer.from_config(qc, pre_quantized=True)
    quantizer.preprocess_model(model, config=config)
    swapped = sum(isinstance(m, (FP8Linear, FP8Experts)) for m in model.modules()) - before

    if embeddings:
        FP8Embedding = getattr(finegrained_fp8, "FP8Embedding", None)
        if FP8Embedding is None:
            raise RuntimeError(
                "the checkpoint stores an embedding table as fp8 with a "
                "per-tensor scale, which needs FP8Embedding from the fork")
        for fqn, mod in embeddings:
            parent_name, _, child = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, FP8Embedding(mod.num_embeddings, mod.embedding_dim,
                                                padding_idx=mod.padding_idx))
            swapped += 1
    return swapped


def set_experts_implementation(model, name):
    """Select transformers' experts implementation for the next export."""
    config = model.config
    config._experts_implementation = name
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    text._experts_implementation = name
