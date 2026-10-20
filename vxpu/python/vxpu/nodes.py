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

"""Dataflow nodes as native packages: what a node server runs (#394).

A node of the decode step's dataflow graph (dataflow.py) is a segment of
the split graph. A dense node is torch ops and Triton kernels, which
AOTInductor already compiles (aoti.py). A host node (the routed experts,
an n-gram lookup) calls a vxpu op whose body is Python: it finds the
engine's banks or tables and asks the plan which implementation to run.
Here that call is replaced by the op's native lowering, so the node is
torch ops and native kernels only and compiles to a package as well:

  vxpu::moe_experts   vxpu_native::fp4_moe_decode, the CPU kernels' fused
                      decode step, on the layer's banks
  vxpu::engram_rows   vxpu_native::gather_rows on the table and its
                      scales, dequantized with torch ops

The tensors the lowering reads (the banks, the tables) are the node's
residents: they become arguments after the node's inputs, passed by
address on every call, never copied into the package. The package is
cached like a dense segment's, under a key of the node's code, its
arguments' shapes and the native kernels' source.

The lowerings are the implementations the engine selects for a decode
step on the CPU (ops.py: cpu-fused, host-parallel), with the same kernels
on the same arguments, so a native node is bit-identical to the node run
in the engine. A node server (next) loads the package and runs it on the
messages its mailboxes receive, with no Python in between.
"""

import hashlib
import inspect

import torch

from . import kernels

_registered = False


def _register_fakes():
    """Shapes for the native ops, so torch.export can trace a node that
    calls them (the ops themselves come with the kernels library)."""
    global _registered
    if _registered:
        return
    if kernels.load() is None:
        raise RuntimeError(f"native nodes need the CPU kernels: {kernels.why_not()}")

    @torch.library.register_fake("vxpu_native::fp4_moe_decode")
    def _(x_in, weights, indices, w1, s1, w3, s3, w2, s2, swiglu_limit):
        return x_in.new_empty((x_in.shape[0], x_in.shape[1]), dtype=torch.float32)

    @torch.library.register_fake("vxpu_native::gather_rows")
    def _(table, indices):
        return table.new_empty(tuple(indices.shape) + (table.shape[1],))

    _registered = True


# ---------------------------------------------------------------------------
# the lowerings: (the host op's arguments, its module) -> (a function of the
# op's tensor arguments and the residents, the residents)

def _lower_moe_experts(moe):
    # the scales as their bytes: the native op reads them so (an e8m0
    # tensor's type cannot be named in AOTInductor's CPU code)
    residents = [moe.w1, moe.w1_scale.view(torch.uint8), moe.w3, moe.w3_scale.view(torch.uint8),
                 moe.w2, moe.w2_scale.view(torch.uint8)]
    start, limit = int(moe.experts_start_idx), float(moe.swiglu_limit)

    def fn(x_in, weights, indices, w1, s1, w3, s3, w2, s2):
        # experts_cpu.forward_batched, fused: the selection relative to the
        # layer's first expert, the weights in fp32
        sel = (indices - start).to(torch.int64)
        return torch.ops.vxpu_native.fp4_moe_decode(x_in, weights.float(), sel, w1, s1, w3, s3, w2, s2, limit)
    return fn, residents


def _scale_float(scale):
    """A scale gathered from the table, as float32: e8m0 scales come as
    their bytes (AOTInductor's CPU code cannot name the e8m0 type) and are
    exactly 2^(byte - 127), the byte in a float's exponent field; 0xFF is
    e8m0's NaN."""
    if scale.dtype != torch.uint8:
        return scale.float()
    bits = (scale.to(torch.int32) << 23).view(torch.float32)
    bits = torch.where(scale == 0xFF, torch.full_like(bits, float("nan")), bits)
    return torch.where(scale == 0, torch.full_like(bits, 2.0 ** -127), bits)


def _lower_engram_rows(embed):
    scale = embed.scale.view(torch.uint8) if embed.scale.dtype == torch.float8_e8m0fnu else embed.scale
    residents = [embed.weight, scale]
    start, end, block = int(embed.vocab_start_idx), int(embed.vocab_end_idx), int(embed.block_size)

    def fn(indices, weight, scale):
        # ops._engram_parallel: the module's lookup, its gathers native
        mask = (indices < start) | (indices >= end)
        local = (indices - start).masked_fill(mask, 0)
        values = torch.ops.vxpu_native.gather_rows(weight, local)
        scales = torch.ops.vxpu_native.gather_rows(scale, local)
        values = values.float().unflatten(-1, (-1, block)) * _scale_float(scales).unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).masked_fill(mask.unsqueeze(-1), 0)
    return fn, residents


# host op target -> (which of external's collections its integer argument
# names a module of, the lowering, the dimension of each of the op's tensor
# arguments that counts tokens: one package serves a decode step's one
# token and a prefill's many)
LOWERINGS = {
    "vxpu.moe_experts.default": ("experts", _lower_moe_experts, (0, 0, 0)),
    "vxpu.engram_rows.default": ("engram", _lower_engram_rows, (1,)),
}


def _host_op(target):
    """The host op a call's target is (its name in LOWERINGS), or None:
    the dispatcher's op, or its Python function when the segment calls it
    directly (segments.call_host_ops_directly)."""
    from . import ops
    direct = {ops.moe_experts._init_fn: "vxpu.moe_experts.default", ops.engram_rows._init_fn: "vxpu.engram_rows.default"}
    name = direct.get(target, str(target))
    return name if name in LOWERINGS else None


def host_call(sub):
    """(the External collection, the module's index) a host node's op
    reads: ("experts", 3) for layer 3's experts."""
    calls = [n for n in sub.graph.nodes if n.op == "call_function" and _host_op(n.target)]
    if len(calls) != 1:
        raise ValueError(f"a native node calls one host op, this one {len(calls)}")
    kind = LOWERINGS[_host_op(calls[0].target)][0]
    return kind, next(a for a in calls[0].args if isinstance(a, int))


class NativeNode(torch.nn.Module):
    """A host node with its host op lowered: called with the node's
    inputs, then its residents."""

    def __init__(self, sub, external):
        super().__init__()
        _register_fakes()
        self.sub = sub
        calls = [n for n in sub.graph.nodes if n.op == "call_function" and _host_op(n.target)]
        if len(calls) != 1:
            raise ValueError(f"a native node calls one host op, this one {len(calls)}")
        call = self.call = calls[0]
        kind, lower, self.token_dims = LOWERINGS[_host_op(call.target)]
        index = next(a for a in call.args if isinstance(a, int))
        module = getattr(external, kind)[index]
        self.fn, self.residents = lower(module)
        self.n_inputs = len([n for n in sub.graph.nodes if n.op == "placeholder"])
        self.description = f"{_host_op(call.target)}({index}) via {lower.__name__}"

    def forward(self, *args):
        inputs, residents = args[:self.n_inputs], args[self.n_inputs:]
        node = self

        class Lowered(torch.fx.Interpreter):
            def call_function(self, target, fargs, kwargs):
                if _host_op(target):
                    tensors = [a for a in fargs if isinstance(a, torch.Tensor)]
                    return node.fn(*tensors, *residents)
                return super().call_function(target, fargs, kwargs)

        return Lowered(self.sub).run(*inputs)

    def cache_extra(self):
        """What the package depends on besides the node's code and shapes:
        the lowering and the native kernels' source."""
        h = hashlib.sha256(self.description.encode())
        h.update(inspect.getsource(inspect.getmodule(NativeNode)).encode())
        for source in kernels.SOURCES:
            with open(source, "rb") as f:
                h.update(f.read())
        return h.hexdigest()


def native_package(packages, sub, external, inputs):
    """The package file of a host node (built first if not cached), for
    any number of tokens, and its residents: run it on its inputs, then
    its residents. inputs: the node's decode-step inputs (their shapes)."""
    node = NativeNode(sub, external)
    # exported on two tokens, with the token dimension dynamic: one token
    # would be specialized (torch.export fixes dimensions of size 1)
    tokens = torch.export.Dim("tokens", min=1, max=1 << 16)
    examples, spec = [], []
    for t, dim in zip(inputs, node.token_dims):
        shape = list(t.shape)
        shape[dim] = 2
        examples.append(torch.zeros(shape, dtype=t.dtype))
        spec.append({dim: tokens})
    args = examples + list(node.residents)
    dynamic = {"args": tuple(spec) + (None,) * len(node.residents)}
    return packages.path(node, args, extra=node.cache_extra() + "tokens", code=sub.code,
                         dynamic_shapes=dynamic), node.residents


def decode_split(artifact_dir):
    """An artifact's decode graph split at its host ops, as the engine
    splits it (its nodes' names and graphs), loaded without weights."""
    import os
    from . import ops, segments  # noqa: F401  (ops registers the vxpu ops the graph calls)
    program = torch.export.load(os.path.join(artifact_dir, "decode.pt2"))
    split, _ = segments.split_at_host_ops(program.module())
    return split


def host_calls(split):
    """{node name: (collection, index)} for a split graph's host nodes."""
    from . import segments
    return {name: host_call(sub) for name, sub in segments.segments_of(split) if is_native(sub)}


def is_native(sub):
    """Whether a segment is a host node this module can lower."""
    return any(n.op == "call_function" and _host_op(n.target) for n in sub.graph.nodes)


