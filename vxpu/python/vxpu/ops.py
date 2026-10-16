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

Each engine owns what it placed (an External) and makes it the ops'
source only while it runs a graph; the exporter only records the shape
information tracing needs.
"""

import contextlib
import threading

import torch
from torch.library import custom_op

from . import experts_cpu


class External:
    """The tensors one engine placed outside its graphs: the fork's MoE
    module per layer (holding that layer's banks) and the fork's n-gram
    table module per engram layer. Owned by the engine, so they go when
    it goes, and active only while that engine runs a graph (see bound):
    two engines in one process, or a load superseded while in flight,
    never see each other's banks."""

    def __init__(self):
        self.experts = {}
        self.engram = {}


# layer id -> row width, for shape inference while tracing (set by the exporter)
ENGRAM_DIM = {}

_active = threading.local()


@contextlib.contextmanager
def bound(external):
    """Make `external` what the ops answer from, in this thread, for the
    duration of the block (an engine wraps each graph call)."""
    previous = getattr(_active, "external", None)
    _active.external = external
    try:
        yield
    finally:
        _active.external = previous


def _current(what, key):
    ext = getattr(_active, "external", None)
    table = getattr(ext, what, None) if ext is not None else None
    if table is None or key not in table:
        raise RuntimeError(f"vxpu::{what}: no {what} bound for {key} (the engine binds the artifact's "
                           "external tensors and runs its graphs under ops.bound)")
    return table[key]


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
    moe = _current("experts", layer)
    host = moe.w1.device
    x, w, i = x_in.to(host), weights.to(host), indices.to(host)
    if experts_cpu.applicable(moe):
        # the fused fp4 kernels (csrc/): same routings, no dequantized copy
        y = experts_cpu.forward_batched(moe, x, w, i) if x.shape[0] == 1 else experts_cpu.forward_grouped(moe, x, w, i)
    elif x.shape[0] == 1:
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
    embed = _current("engram", table)
    host = embed.weight.device
    return embed.lookup(indices.to(host)).to(indices.device)


@engram_rows.register_fake
def _(indices, table):
    return indices.new_empty(tuple(indices.shape) + (ENGRAM_DIM[table],), dtype=torch.bfloat16)
