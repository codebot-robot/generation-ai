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
model the way the loader would (``apply_fp8``), choose the experts
implementation per graph (``grouped_mm`` for prefill, ``batched_mm`` for
decode: both static-shape and the same routing computation transformers
runs), and tell the executor how to assemble a stacked expert parameter
from the checkpoint's per-expert tensors (``stacked_sources``).
"""

import re


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


def _lookup(manifest_tensors, fqn):
    """Find a checkpoint tensor for a module path, peeling wrapper prefixes."""
    parts = fqn.split(".")
    for i in range(len(parts)):
        if ".".join(parts[i:]) in manifest_tensors:
            return True
    return False


def apply_fp8(model, config, manifest_tensors=None):
    """Install transformers' fp8 modules on a (meta) model, as its loader does.

    Runs the same HfQuantizer preprocessing ``from_pretrained`` runs. The
    loader additionally lets the *checkpoint* decide: a linear whose
    weight has no ``weight_scale_inv`` beside it (Qwen's bf16 lm_head,
    with an empty ``modules_to_not_convert``) stays a plain linear. Given
    the manifest, the same rule is applied here before the swap.
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
    qc = config.quantization_config
    qc = FineGrainedFP8Config.from_dict(qc) if isinstance(qc, dict) else qc
    skip = list(qc.modules_to_not_convert or [])
    if manifest_tensors:
        for fqn, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                probe = fqn
            elif isinstance(getattr(mod, "gate_up_proj", None), nn.Parameter):
                probe = f"{fqn}.0.gate_proj"  # the checkpoint stores experts one by one
            else:
                continue
            if (_lookup(manifest_tensors, probe + ".weight")
                    and not _lookup(manifest_tensors, probe + ".weight_scale_inv")):
                skip.append(fqn)
    qc.modules_to_not_convert = sorted(set(skip))
    before = sum(isinstance(m, (FP8Linear, FP8Experts)) for m in model.modules())
    quantizer = AutoHfQuantizer.from_config(qc, pre_quantized=True)
    quantizer.preprocess_model(model, config=config)
    return sum(isinstance(m, (FP8Linear, FP8Experts)) for m in model.modules()) - before


def set_experts_implementation(model, name):
    """Select transformers' experts implementation for the next export."""
    config = model.config
    config._experts_implementation = name
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    text._experts_implementation = name


EXPERTS_PARAM = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<which>gate_up_proj|down_proj)(?P<scale>_scale_inv)?$")


def stacked_sources(fqn, manifest_tensors, num_experts):
    """For a stacked expert parameter, the checkpoint tensors it is
    assembled from: a list (over experts) of lists (concatenated along
    dim 0) of manifest tensor names. None if ``fqn`` is not one.

    Matching peels wrapper prefixes the way the exporter does elsewhere.
    """
    m = EXPERTS_PARAM.match(fqn)
    if not m:
        return None
    suffix = ".weight" + (m.group("scale") or "")
    projs = ["gate_proj", "up_proj"] if m.group("which") == "gate_up_proj" else ["down_proj"]
    parts = m.group("prefix").split(".")
    for start in range(len(parts)):
        prefix = ".".join(parts[start:])
        names = [[f"{prefix}.{i}.{p}{suffix}" for p in projs] for i in range(num_experts)]
        if all(n in manifest_tensors for row in names for n in row):
            return names
    return None
