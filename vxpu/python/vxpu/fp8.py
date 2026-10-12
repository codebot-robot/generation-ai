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

"""Block-scaled FP8 checkpoints (``quant_method: fp8``) as aten-only graphs.

Model makers ship checkpoints whose linear weights are ``float8_e4m3fn``
with one scale per ``block_size`` (typically 128x128) block
(``<name>.weight`` + ``<name>.weight_scale_inv``; the scales are float32
or, in Qwen's releases, bf16 — they are bound as stored). transformers runs them
through GPU-only kernels; here the same tensors are consumed by plain
aten ops so the graph exports on the meta device and runs wherever
torch runs: the weight is dequantized block-wise *during* the matmul,
never materialized in full up front.

Two module replacements, applied on the meta model before export:

* ``FP8Linear`` — ``nn.Linear`` whose parameters are the checkpoint's
  ``weight`` (fp8) and ``weight_scale_inv``; forward dequantizes and
  calls ``F.linear``.
* ``FP8Experts`` — a mixture-of-experts expert bank whose stacked
  parameters hold every expert's fp8 weights and scales. The checkpoint
  stores experts one by one (``experts.<i>.gate_proj.weight`` ...); the
  binding's ``stacked`` category tells the executor how to assemble
  them. Decode gathers only the selected experts' blocks and
  dequantizes those; prefill dequantizes the layer's experts once and
  runs them densely, so its cost is bounded whatever the prompt length.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn


def dequantize_block(weight, scale_inv, block_size, dtype=torch.bfloat16):
    """fp8 [.., N, K] with scales [.., N/bn, K/bk] -> dtype [.., N, K]."""
    bn, bk = block_size
    *lead, n, k = weight.shape
    w = weight.to(dtype).reshape(*lead, n // bn, bn, k // bk, bk)
    s = scale_inv.to(dtype).reshape(*lead, n // bn, 1, k // bk, 1)
    return (w * s).reshape(*lead, n, k)


class FP8Linear(nn.Module):
    """``F.linear`` over a block-scaled fp8 weight (checkpoint names)."""

    def __init__(self, in_features, out_features, block_size, bias=False,
                 scale_dtype=torch.float32):
        super().__init__()
        bn, bk = block_size
        self.in_features, self.out_features = in_features, out_features
        self.block_size = tuple(block_size)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8), requires_grad=False)
        self.weight_scale_inv = nn.Parameter(
            torch.empty((out_features + bn - 1) // bn, (in_features + bk - 1) // bk,
                        dtype=scale_dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None

    def forward(self, x):
        w = dequantize_block(self.weight, self.weight_scale_inv, self.block_size, x.dtype)
        return F.linear(x, w, self.bias)


class FP8Experts(nn.Module):
    """Expert bank for gated MLP experts (gate/up/down), fp8 block-scaled.

    Parameters (stacked over experts, assembled by the executor):
      gate_up_proj            fp8  [E, 2*I, H]   rows: gate then up
      gate_up_proj_scale_inv  f32|bf16 [E, 2*I/bn, H/bk]
      down_proj               fp8  [E, H, I]
      down_proj_scale_inv     f32|bf16 [E, H/bn, I/bk]

    ``mode`` is fixed at export time: "decode" gathers the selected
    experts per token; "prefill" dequantizes all experts once and
    evaluates every expert for every token, masking by routing weight.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, block_size, act_fn,
                 scale_dtype=torch.float32):
        super().__init__()
        bn, bk = block_size
        self.num_experts, self.hidden_dim, self.intermediate_dim = num_experts, hidden_dim, intermediate_dim
        self.block_size = tuple(block_size)
        self.act_fn = act_fn
        self.mode = "decode"
        e, h, i = num_experts, hidden_dim, intermediate_dim
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h, dtype=FP8), requires_grad=False)
        self.gate_up_proj_scale_inv = nn.Parameter(
            torch.empty(e, (2 * i + bn - 1) // bn, (h + bk - 1) // bk, dtype=scale_dtype), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(e, h, i, dtype=FP8), requires_grad=False)
        self.down_proj_scale_inv = nn.Parameter(
            torch.empty(e, (h + bn - 1) // bn, (i + bk - 1) // bk, dtype=scale_dtype), requires_grad=False)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.mode == "prefill":
            return self._dense(hidden_states, top_k_index, top_k_weights)
        return self._gather(hidden_states, top_k_index, top_k_weights)

    def _gather(self, x, top_k_index, top_k_weights):
        """Selected experts only: S = tokens * top_k (expert, token) pairs."""
        t, k = top_k_index.shape
        idx = top_k_index.reshape(-1)                                   # [S]
        xs = x.repeat_interleave(k, dim=0)                              # [S, H]
        w1 = dequantize_block(self.gate_up_proj[idx], self.gate_up_proj_scale_inv[idx],
                              self.block_size, x.dtype)                 # [S, 2I, H]
        gate, up = torch.bmm(w1, xs.unsqueeze(-1)).squeeze(-1).chunk(2, dim=-1)
        h = self.act_fn(gate) * up                                      # [S, I]
        w2 = dequantize_block(self.down_proj[idx], self.down_proj_scale_inv[idx],
                              self.block_size, x.dtype)                 # [S, H, I]
        out = torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1)                # [S, H]
        out = out * top_k_weights.reshape(-1, 1).to(out.dtype)
        return out.reshape(t, k, -1).sum(dim=1)

    def _dense(self, x, top_k_index, top_k_weights):
        """Every expert for every token; routing weights as a dense mask."""
        t = x.shape[0]
        w1 = dequantize_block(self.gate_up_proj, self.gate_up_proj_scale_inv, self.block_size, x.dtype)
        gu = torch.matmul(w1, x.t())                                    # [E, 2I, T]
        gate, up = gu.chunk(2, dim=1)
        h = self.act_fn(gate) * up                                      # [E, I, T]
        w2 = dequantize_block(self.down_proj, self.down_proj_scale_inv, self.block_size, x.dtype)
        y = torch.matmul(w2, h)                                         # [E, H, T]
        dense = torch.zeros(t, self.num_experts, dtype=x.dtype, device=x.device)
        dense = dense.scatter_add(1, top_k_index, top_k_weights.to(x.dtype))  # [T, E]
        return torch.einsum("eht,te->th", y, dense)


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


def apply_fp8(model, config, scale_dtype=torch.float32):
    """Swap Linear/expert modules of a meta model for their fp8 forms.

    Mirrors transformers' replacement rules: every ``nn.Linear`` not in
    ``modules_to_not_convert`` (nor ``lm_head``) becomes ``FP8Linear``;
    an expert bank (a module with stacked ``gate_up_proj``/``down_proj``
    parameters) becomes ``FP8Experts``. ``scale_dtype`` is the dtype the
    checkpoint stores ``weight_scale_inv`` in (bf16 in Qwen's exports,
    float32 in others); the graph binds those tensors as they are.
    Returns the count of swaps.
    """
    settings = fp8_config(config)
    if settings is None:
        return 0
    block, skip = settings["block_size"], settings["modules_to_not_convert"]
    swapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            fqn = f"{name}.{child_name}" if name else child_name
            if fqn in skip or any(fqn.endswith("." + s) or fqn == s for s in skip) or child_name == "lm_head":
                continue
            if isinstance(child, nn.Linear):
                new = FP8Linear(child.in_features, child.out_features, block,
                                bias=child.bias is not None, scale_dtype=scale_dtype)
            elif (hasattr(child, "gate_up_proj") and hasattr(child, "down_proj")
                  and isinstance(getattr(child, "gate_up_proj"), nn.Parameter)):
                e, two_i, h = child.gate_up_proj.shape
                new = FP8Experts(e, h, two_i // 2, block, child.act_fn, scale_dtype=scale_dtype)
            else:
                continue
            setattr(module, child_name, new)
            swapped += 1
    return swapped


def scale_dtype_of(manifest_tensors):
    """The dtype the checkpoint stores its fp8 block scales in."""
    for name, ref in manifest_tensors.items():
        if name.endswith("_scale_inv"):
            return {"F32": torch.float32, "BF16": torch.bfloat16,
                    "F16": torch.float16}[ref["dtype"]]
    return torch.float32


def set_experts_mode(model, mode):
    for m in model.modules():
        if isinstance(m, FP8Experts):
            m.mode = mode


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
