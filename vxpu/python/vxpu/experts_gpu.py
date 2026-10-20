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


"""Hot experts on the GPUs, computed alongside the cold ones on the CPU.

A decode step's dense work (GPU) and routed experts (CPU) cannot overlap
for one sequence: each layer's experts need its attention output, and the
next layer needs theirs. But a layer's experts can be split. Expert use is
skewed (#317: on real traffic the hottest quarter of each layer's experts
carry ~80% of decode selections), so the hottest experts are kept on the
GPUs, in the checkpoint's packed fp4 with their scales, and computed there
(triton_ops.fp4_experts_decode) while the CPU computes the rest with its
kernels; the partial sums are added. The CPU's share of the step falls to
its share of the selections.

Where each expert lives is a layer's Residency: a pool of slots on each
GPU that holds experts (the engine's own device first, so the hottest
need no copy between devices), and the host banks for everything else.
Every expert is in at most one pool; a pool's slot map says, per expert,
its slot there or -1. The kernels take that map as it is, so a pool on
another GPU runs the same code: the token's activations go there (a few
KB), its partial sum comes back.

Which experts are hot is a placement, from the engine's own statistics
(`vxpu stats --records` into `python -m vxpu.expert_stats
--placement-gb`), named by the plan's gpu_experts choice
(VXPU_GPU_EXPERTS). This is the first placement chosen from the
statistics the engine keeps (#316, #348).
"""

import json

import torch


class Pool:
    """One GPU's slots for one layer: the held experts' banks (packed fp4
    and e8m0 scales, rows gathered from the host banks), which expert each
    slot holds, and the slot of every expert (-1 where it is not held
    here), on the host and on this GPU."""

    def __init__(self, moe, expert_ids, device):
        ids = sorted({int(e) for e in expert_ids})
        idx = torch.tensor(ids, dtype=torch.int64)
        def held(t):
            return t.index_select(0, idx - moe.experts_start_idx).contiguous().to(device)
        self.device = torch.device(device)
        self.w1, self.s1 = held(moe.w1), held(moe.w1_scale)
        self.w3, self.s3 = held(moe.w3), held(moe.w3_scale)
        self.w2, self.s2 = held(moe.w2), held(moe.w2_scale)
        slot = torch.full((moe.n_local_experts,), -1, dtype=torch.int64)
        slot[idx - moe.experts_start_idx] = torch.arange(len(ids))
        self.experts = ids  # slot -> expert
        self.slot_cpu = slot
        self.slot_gpu = slot.to(device)
        self.bytes = sum(t.numel() * t.element_size() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))
        self.stream = _side_stream(self.device)

    @property
    def count(self):
        return len(self.experts)


class Residency:
    """Where one layer's experts live: pools on GPUs, the host banks for
    the rest. held_cpu[e] is True where expert e (local index) is in a
    pool, so the CPU skips it."""

    def __init__(self, moe, by_device):
        self.pools = [Pool(moe, ids, dev) for dev, ids in by_device if ids]
        held = torch.zeros(moe.n_local_experts, dtype=torch.bool)
        for pool in self.pools:
            seen = held & (pool.slot_cpu >= 0)
            if seen.any():
                raise ValueError(f"experts {(seen.nonzero().view(-1) + moe.experts_start_idx).tolist()} "
                                 "are placed on more than one GPU")
            held |= pool.slot_cpu >= 0
        self.held_cpu = held

    @property
    def count(self):
        return sum(p.count for p in self.pools)

    @property
    def bytes(self):
        return sum(p.bytes for p in self.pools)


_streams = {}


def _side_stream(device):
    """One side stream per device for every layer's pool there (layers run
    one after another, so they never contend)."""
    key = str(device)
    if key not in _streams:
        _streams[key] = torch.cuda.Stream(device=device)
    return _streams[key]


def devices_for(engine_device):
    """The GPUs a placement's pools go to, in order: the engine's own
    device, then every other visible GPU by index."""
    first = torch.device(engine_device)
    if first.index is None:
        first = torch.device("cuda", torch.cuda.current_device())
    return [first] + [torch.device("cuda", i) for i in range(torch.cuda.device_count()) if i != first.index]


def load_placement(path):
    """[{layer: [expert ids]}] per GPU, in devices_for's order, from a
    placement file (expert_stats --placement-gb): format
    vxpu-placement/2 lists "gpus"; /1 is one GPU's "layers"."""
    with open(path) as f:
        doc = json.load(f)
    gpus = doc["gpus"] if "gpus" in doc else [doc]
    return [{int(layer): ids for layer, ids in gpu["layers"].items()} for gpu in gpus]
