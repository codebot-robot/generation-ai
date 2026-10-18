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

"""The routed experts as mailboxes, drained by their owners (#372).

A decode step's experts node sends each (token, k) pair the router chose
to the owner of that expert: a message is the token's activation row,
the expert and the routing weight. An owner drains its messages and
answers each with the pair's reply, the expert's down projection rounded
to bf16 (what the model's module adds up). The replies land in
per-(token, k) slots and are summed in k order, into fp32 from zero, as
the fused CPU kernel sums them. So the sum no longer depends on which
owner answered which pair. Only the replies themselves can differ
between owners: an owner on the CPU answers exactly what the fused
kernel would have added, so experts on the CPU, in this process or
another, give bit-identical results. A GPU's kernels round in the same
places but accumulate their dot products in a different order, so a
reply from a GPU can differ in its last bf16 bit.

The owners:

  cpu    the CPU kernels in this process: each pair is a one-expert call
         of the fused decode kernel, whose sum of one term is the term
  pool   a GPU's slot pool (experts_gpu.Pool, #370): its kernels on its
         side stream, launched before the CPU drains its own
  remote another process, here or on another machine, holding some
         layers' banks (expert_server.py): it answers the pairs of those
         layers that no pool holds, with the same CPU kernels, so its
         replies are bit-identical to the local CPU's

Which owner holds an expert is the layer's residency, read on every
call, so experts moving between owners (experts_gpu.Mover, #371) need
nothing more here. Every owner's work is posted before any reply is
waited for, so the GPUs and the CPU drain their mailboxes at the same
time.
"""

import socket

import torch

from . import kernels, triton_ops


class CpuOwner:
    """Answers with the CPU kernels, in this process."""

    name = "cpu"

    def answer(self, moe, x, tokens, experts, weights, layer=None):
        """Replies [m, dim] fp32 for m pairs: x [n, dim] on the CPU,
        tokens/experts/weights [m]."""
        k = kernels.load()
        sel = (experts - moe.experts_start_idx).to(torch.int64)
        return k.fp4_moe_decode(x.index_select(0, tokens).contiguous(), weights.float().view(-1, 1).contiguous(),
                                sel.view(-1, 1).contiguous(), moe.w1, moe.w1_scale, moe.w3, moe.w3_scale,
                                moe.w2, moe.w2_scale, float(moe.swiglu_limit))


class RemoteOwner:
    """Answers from an expert server (expert_server.py) over TCP, with the
    CpuOwner's interface. Every call is bounded by TIMEOUT_S; a call that
    fails on the connection reconnects once and is retried, and then the
    error names the server."""

    TIMEOUT_S = 60

    def __init__(self, address, checkpoint=None):
        self.address = address
        self.name = f"remote {address}"
        self.checkpoint = checkpoint
        self.sock = None
        self.layers = self._connect()

    def _connect(self):
        from . import expert_server as wire
        host, port = self.address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=self.TIMEOUT_S)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wire.send(self.sock, {"op": "hello"})
        info = wire.recv_header(self.sock)
        if self.checkpoint is not None and info["checkpoint"] != self.checkpoint:
            raise ValueError(f"expert server {self.address} holds another checkpoint's experts")
        return set(info["layers"])

    def answer(self, moe, x, tokens, experts, weights, layer=None):
        """Replies [m, dim] fp32 for m pairs of `layer`, from the server."""
        from . import expert_server as wire
        for attempt in (0, 1):
            try:
                if self.sock is None:
                    self._connect()
                wire.send(self.sock, {"op": "answer", "layer": layer, "n": x.shape[0], "m": tokens.numel()},
                          x.to(torch.bfloat16), tokens.to(torch.int32), experts.to(torch.int32),
                          weights.to(torch.float32))
                header = wire.recv_header(self.sock)
                if "error" in header:
                    self.close()  # the server closes after an error
                    raise RuntimeError(f"expert server {self.address}: {header['error']}")
                return wire.recv_tensor(self.sock, torch.bfloat16, (header["m"], moe.dim)).float()
            except (OSError, ConnectionError) as e:  # timeouts included
                self.close()
                if attempt:
                    raise RuntimeError(f"expert server {self.address}: {e}") from e

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class Layer:
    """One layer's mailboxes: the owners of its experts, and the step's
    dispatch, drain and fixed-order combine."""

    def __init__(self, moe, layer, residency=None, remote=None, cpu=None):
        self.moe = moe
        self.layer = layer
        self.residency = residency  # experts_gpu.Residency: the GPU pools and what they hold, or None
        self.remote = remote  # a RemoteOwner holding this layer, answering what no pool holds; or None
        self.cpu = cpu or CpuOwner()

    def forward(self, x_in, weights, indices, host_inputs=None):
        """y [n, dim] fp32 on x_in's device. host_inputs: the same three
        tensors already on the host (the engine's pinned copies), or None
        to copy them here. Returns (y, pairs answered per owner)."""
        moe, dev = self.moe, x_in.device
        n, topk = indices.shape
        current = torch.cuda.current_stream(dev) if dev.type == "cuda" else None
        posted = []
        for pool in (self.residency.pools if self.residency is not None else []):
            if current is not None:  # inputs on a GPU: the dense work before produces them
                pool.stream.wait_stream(current)
            with torch.cuda.stream(pool.stream):
                if pool.device == dev:
                    x, w, i = x_in, weights, indices
                else:  # a few KB to the other GPU
                    x, w, i = (t.to(pool.device, non_blocking=True) for t in (x_in, weights, indices))
                slots = pool.slot_gpu[i - moe.experts_start_idx]
                replies = triton_ops.fp4_experts_decode(x, w, slots, pool.w1, pool.s1, pool.w3, pool.s3,
                                                        pool.w2, pool.s2, moe.swiglu_limit, per_pair=True)
            posted.append((pool, replies))
        x, w, i = host_inputs if host_inputs is not None else (t.to("cpu") for t in (x_in, weights, indices))
        held = self.residency.held_cpu[i - moe.experts_start_idx] if self.residency is not None else None
        cold = ~held if held is not None else torch.ones_like(i, dtype=torch.bool)
        counts = {}
        slots = torch.zeros(n, topk, moe.dim, dtype=torch.float32, device=dev)
        if cold.any():
            tokens = torch.arange(n).view(n, 1).expand(n, topk)[cold]
            owner = self.remote or self.cpu
            answered = owner.answer(moe, x, tokens, i[cold], w[cold], layer=self.layer)
            owner = "remote" if owner is self.remote else owner.name
            slots[cold.to(dev)] = answered.to(dev, non_blocking=True)
            counts[owner] = int(cold.sum())
        for pool, replies in posted:
            if pool.device != dev:  # to the inputs' device: another GPU, or the host for a CPU-placed node
                with torch.cuda.stream(pool.stream):
                    replies = replies.to(dev, non_blocking=current is not None)
            if current is not None:
                current.wait_stream(pool.stream)
                replies.record_stream(current)
            mine = (pool.slot_cpu[i - moe.experts_start_idx] >= 0).to(dev)
            slots = torch.where(mine.unsqueeze(-1), replies, slots)
            counts[str(pool.device)] = int(mine.sum())
        y = torch.zeros(n, moe.dim, dtype=torch.float32, device=dev)
        for k in range(topk):  # in k order, from zero: the fused kernel's sum
            y = y + slots[:, k]
        return y, counts
