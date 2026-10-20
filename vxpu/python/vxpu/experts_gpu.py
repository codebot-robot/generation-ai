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


"""Hot experts on the GPU, computed alongside the cold ones on the CPU.

A decode step's dense work (GPU) and routed experts (CPU) cannot overlap
for one sequence: each layer's experts need its attention output, and the
next layer needs theirs. But a layer's experts can be split. Expert use is
skewed (#317: on real traffic the hottest quarter of each layer's experts
carry ~80% of decode selections), so the hottest experts are kept on the
GPU, in the checkpoint's packed fp4 with their scales, and computed there
(triton_ops.fp4_experts_decode) while the CPU computes the rest with its
kernels; the two partial sums are added. The CPU's share of the step falls
to its share of the selections.

Which experts are hot is a placement: {layer: [expert ids]}, from the
engine's own statistics (`vxpu stats --records` into `python -m
vxpu.expert_stats --placement-gb`), named by the plan's gpu_experts choice
(VXPU_GPU_EXPERTS). This is the first placement chosen from the statistics
the engine keeps (#316, #348).
"""

import json

import torch


class HotExperts:
    """One layer's hot experts held on the GPU: their banks (packed fp4 and
    e8m0 scales, rows gathered from the host banks) and the slot of every
    expert (-1 where it stays on the host), on both devices."""

    def __init__(self, moe, expert_ids, device):
        ids = sorted({int(e) for e in expert_ids})
        idx = torch.tensor(ids, dtype=torch.int64)
        def held(t):
            return t.index_select(0, idx).contiguous().to(device)
        self.w1, self.s1 = held(moe.w1), held(moe.w1_scale)
        self.w3, self.s3 = held(moe.w3), held(moe.w3_scale)
        self.w2, self.s2 = held(moe.w2), held(moe.w2_scale)
        slot = torch.full((moe.n_local_experts,), -1, dtype=torch.int64)
        slot[idx - moe.experts_start_idx] = torch.arange(len(ids))
        self.slot_cpu = slot
        self.slot_gpu = slot.to(device)
        self.count = len(ids)
        self.bytes = sum(t.numel() * t.element_size() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))
        self.stream = _side_stream(device)


_streams = {}


def _side_stream(device):
    """One side stream per device for every layer's hot experts (layers run
    one after another, so they never contend)."""
    key = str(device)
    if key not in _streams:
        _streams[key] = torch.cuda.Stream(device=device)
    return _streams[key]


def load_placement(path):
    """{layer: [expert ids]} from a placement file (expert_stats --placement-gb)."""
    with open(path) as f:
        doc = json.load(f)
    return {int(layer): ids for layer, ids in doc["layers"].items()}
