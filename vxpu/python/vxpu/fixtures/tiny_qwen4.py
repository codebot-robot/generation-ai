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

"""A tiny random Qwen4 (Qwen3.8-style hybrid) text checkpoint: Gated
DeltaNet linear attention, indexed attention, a per-layer n-gram
embedding, MoE experts. Covers the export wrapper's linear-attention
state handling and the model's static-cache path in the fork, without a
download. Requires transformers >= 5.18.

    python -m vxpu.fixtures.tiny_qwen4 OUT_DIR
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM


def config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    return Qwen4ExpTextConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        num_experts=4, num_experts_per_tok=2,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32,
        indexer_budget=8, indexer_compress_ratio=4,
        ple_layer_ids=[1], ple_embed_dim=128, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=20_000,
        split_ngram_parts=4, hc_count=4, hc_lowrank=16,
        max_position_embeddings=256, dtype=torch.bfloat16,
        eos_token_id=2, pad_token_id=0)


def write(out_dir, seed=0):
    torch.manual_seed(seed)
    cfg = config()
    cfg._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(cfg)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"do_sample": False}, f)
    return out_dir


if __name__ == "__main__":
    print(write(sys.argv[1]))
