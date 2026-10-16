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

"""Custom ops for the tensors a graph does not hold.

A model larger than its device keeps its biggest tensors outside the
graph: the mixture-of-experts banks (272 GB of fp4 for
DeepSeek-V4.1-Flash) and the n-gram tables (2 x 98 GB). The exporter
traces calls to the two ops below in their place, so the artifact's
graphs carry the dense model only, and the executor answers the calls
from wherever it placed those tensors: host memory, a memory-mapped
file on local SSD, or another device. The math is the fork's own code
on the same bytes, so the result does not depend on the placement.

The registries are per process: the engine fills them when it loads a
model, the exporter fills the shape information it needs to trace.
"""

import torch
from torch.library import custom_op

# layer id -> the fork's MoE module holding that layer's banks (host side)
EXPERTS = {}
# layer id -> the fork's ParallelEngramEmbedding holding that layer's table
ENGRAM = {}
# layer id -> row width, for shape inference while tracing
ENGRAM_DIM = {}


@custom_op("vxpu::moe_experts", mutates_args=())
def moe_experts(x_in: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                layer: int) -> torch.Tensor:
    """The routed experts of `layer` applied to x_in [n, dim] (already
    input-quantized, as the module does), with gate weights and expert
    indices [n, topk]. Returns y [n, dim] float32, what the module's own
    routings return. One token takes the batched routing (gather the
    chosen experts), more take the grouped one (sorted by expert), the
    same choice the exporter made per graph."""
    if x_in.is_meta:  # a graph walked on the meta device (loading, device passes): shape only
        return x_in.new_empty((x_in.shape[0], x_in.shape[1]), dtype=torch.float32)
    moe = EXPERTS.get(layer)
    if moe is None:
        raise RuntimeError(f"vxpu::moe_experts: layer {layer} has no experts bound (the engine "
                           "binds the artifact's external tensors before running a graph)")
    host = moe.w1.device
    x, w, i = x_in.to(host), weights.to(host), indices.to(host)
    if x.shape[0] == 1:
        y = moe._forward_batched(x, w, i)
    else:
        y = moe._forward_grouped(x, w, i)
    return y.to(x_in.device)


@moe_experts.register_fake
def _(x_in, weights, indices, layer):
    return x_in.new_empty((x_in.shape[0], x_in.shape[1]), dtype=torch.float32)


@custom_op("vxpu::engram_rows", mutates_args=())
def engram_rows(indices: torch.Tensor, table: int) -> torch.Tensor:
    """The rows of n-gram table `table` for hash ids `indices`, dequantized
    to bf16 [..., dim], what the module's lookup returns."""
    if indices.is_meta:
        return indices.new_empty(tuple(indices.shape) + (ENGRAM_DIM[table],), dtype=torch.bfloat16)
    embed = ENGRAM.get(table)
    if embed is None:
        raise RuntimeError(f"vxpu::engram_rows: table {table} is not bound")
    host = embed.weight.device
    return embed.lookup(indices.to(host)).to(indices.device)


@engram_rows.register_fake
def _(indices, table):
    return indices.new_empty(tuple(indices.shape) + (ENGRAM_DIM[table],), dtype=torch.bfloat16)


def reset():
    """Forget every bound tensor (tests, or a model unloaded)."""
    EXPERTS.clear()
    ENGRAM.clear()
