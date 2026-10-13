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

"""A tiny random Qwen3-MoE checkpoint quantized the way model makers
ship fp8 (block scales, per-expert tensors, bf16 scales), written as a
local checkpoint directory. Covers the fp8 linear, the expert bank
assembled from per-expert tensors, and the checkpoint-driven skip of an
unquantized lm_head, without a download.

    python -m vxpu.fixtures.tiny_fp8_moe OUT_DIR
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

BLOCK = 128
FP8 = torch.float8_e4m3fn


def config():
    return Qwen3MoeConfig(
        vocab_size=512, hidden_size=256, intermediate_size=512,
        moe_intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        num_experts=8, num_experts_per_tok=2, dtype=torch.bfloat16,
        tie_word_embeddings=False,
        quantization_config={"quant_method": "fp8",
                             "weight_block_size": [BLOCK, BLOCK],
                             "activation_scheme": "dynamic",
                             # lm_head is left unquantized in the checkpoint;
                             # the gates are too small to be worth it.
                             "modules_to_not_convert": [
                                 "lm_head", "model.layers.0.mlp.gate",
                                 "model.layers.1.mlp.gate"]})


def quantize(w):
    """Block-wise symmetric fp8: value = fp8 * scale_inv, scales stored bf16."""
    n, k = w.shape
    blocks = w.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = 448.0 / amax
    q = (blocks * scale).clamp(-448, 448).reshape(n, k).to(FP8)
    return q, (1.0 / scale).reshape(n // BLOCK, k // BLOCK).to(torch.bfloat16)


def write(out_dir, seed=0):
    torch.manual_seed(seed)
    cfg = config()
    model = AutoModelForCausalLM.from_config(cfg)
    tensors = {}
    for name, p in model.named_parameters():
        w = p.detach()
        if ".mlp.experts." in name:
            # Stacked in the module, one tensor per expert in the checkpoint.
            prefix, which = name.rsplit(".", 1)
            projs = ["gate_proj", "up_proj"] if which == "gate_up_proj" else ["down_proj"]
            rows = w.shape[1] // len(projs)
            for i in range(w.shape[0]):
                for j, proj in enumerate(projs):
                    q, s = quantize(w[i, j * rows:(j + 1) * rows])
                    tensors[f"{prefix}.{i}.{proj}.weight"] = q
                    tensors[f"{prefix}.{i}.{proj}.weight_scale_inv"] = s
        elif name.endswith(".weight") and w.dim() == 2 and "self_attn" in name:
            q, s = quantize(w)
            tensors[name] = q
            tensors[name + "_scale_inv"] = s
        else:
            tensors[name] = w.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    cfg.save_pretrained(out_dir)
    # generation_config: greedy, no special stopping
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"do_sample": False}, f)
    return out_dir


if __name__ == "__main__":
    print(write(sys.argv[1]))
