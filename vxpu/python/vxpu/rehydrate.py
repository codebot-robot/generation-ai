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

"""Rehydrate a vXPU artifact into runnable modules.

bound tensors are fetched by HTTP range request, one per tensor,
streaming straight to the target device (flat memory profile; warm
reloads come from the server's digest keep-alive). derived
tensors are recomputed from config using transformers' own formula
registry (exact RoPE for every rope_type, including per-layer-type
hybrids). state tensors are zero-initialized per session.

The graphs were traced on the meta device; after placement they are
retargeted to the executor's device, and device-specific compat passes
are applied (e.g. histc(int) has no CPU kernel; for expert counting it
is exactly bincount).
"""

import hashlib
import math
import os

import requests
import torch
from torch.export.passes import move_to_device_pass

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def fetch_tensor(ref, files, cas_dir=None):
    """One ranged read per tensor.

    With cas_dir set, bytes are cached on disk under a content-derived
    key (for persistent volumes). Without it, bytes stream straight to
    the caller — one tensor in flight, flat memory profile; warm
    reloads come from the engine keep-alive (digest reuse), not disk.

    Returns (tensor, bytes_downloaded); 0 when served from cache.
    """
    def download():
        url = files[ref["file_sha256"]]["source"]
        end = ref["offset"] + ref["length"] - 1
        resp = requests.get(
            url, headers={"Range": f"bytes={ref['offset']}-{end}"})
        resp.raise_for_status()
        if resp.status_code != 206:
            raise RuntimeError(
                f"expected HTTP 206 Partial Content, got {resp.status_code}")
        content = resp.content
        if len(content) != ref["length"]:
            raise RuntimeError(
                f"expected {ref['length']} bytes, got {len(content)} bytes")
        return content

    downloaded = 0
    if cas_dir is None:
        data = bytearray(download())
        downloaded = ref["length"]
    else:
        key = hashlib.sha256(
            f"{ref['file_sha256']}:{ref['offset']}:{ref['length']}"
            .encode()).hexdigest()
        path = os.path.join(cas_dir, key)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(download())
            downloaded = ref["length"]
        with open(path, "rb") as f:
            data = bytearray(f.read())
    tensor = torch.frombuffer(
        data, dtype=DTYPES[ref["dtype"]]).view(ref["shape"])
    return tensor, downloaded


def derived_tensor(fqn, meta, config):
    """Recompute a config-derived tensor to the shape the graph expects.

    RoPE tables use transformers' own ROPE_INIT_FUNCTIONS registry,
    keyed by rope_type and (for hybrid-attention models) layer type —
    exact by construction. Embedding scales are sqrt(hidden).
    """
    text_config = (config.get_text_config()
                   if hasattr(config, "get_text_config") else config)
    shape = tuple(meta.shape)
    dtype = (meta.dtype if meta.dtype.is_floating_point
             else torch.float32)

    if "embed_scale" in fqn:
        hidden = text_config.hidden_size
        if "per_layer" in fqn and getattr(
                text_config, "hidden_size_per_layer_input", 0):
            hidden = text_config.hidden_size_per_layer_input
        return torch.full(shape or (1,), math.sqrt(hidden),
                          dtype=dtype).reshape(shape)

    # Mirror transformers' own rope init (modeling_rope_utils + the
    # per-model rotary modules): select the layer's rope_type, call the
    # registered init fn, and pass the exact kwargs transformers passes
    # (e.g. head_dim_key="global_head_dim" for Gemma 4 proportional
    # full-attention RoPE). Reproducing this rather than re-deriving
    # formulas keeps hybrid/partial-rotary models exact.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    layer_type = None
    for candidate in ("sliding_attention", "full_attention"):
        if candidate in fqn:
            layer_type = candidate
    rope_params = getattr(text_config, "rope_parameters", None) or {}
    if layer_type and isinstance(rope_params.get(layer_type), dict):
        params = rope_params[layer_type]
    else:
        params = rope_params if isinstance(rope_params, dict) else {}
    rope_type = params.get("rope_type", "default")

    kwargs = {}
    if layer_type:
        kwargs["layer_type"] = layer_type
        if layer_type == "full_attention" and rope_type == "proportional":
            kwargs["head_dim_key"] = "global_head_dim"
    init_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
    if init_fn is not None:
        inv_freq, _ = init_fn(text_config, **kwargs)
        want = shape[-1] if shape else inv_freq.numel()
        if inv_freq.numel() == want:
            return inv_freq.to(dtype).reshape(shape)
    # Fallback: plain geometric series over the dims the graph expects.
    length = shape[-1] if shape else 32
    theta = params.get("rope_theta",
                       getattr(text_config, "rope_theta", 10000.0))
    exponent = torch.arange(0, length, dtype=torch.float32) / max(length, 1)
    return (1.0 / (theta ** exponent)).to(dtype).reshape(shape)


def cpu_compat(exported):
    """Rewrite ops whose CPU kernels lack integer support.

    histc(int) has no CPU kernel; for integer inputs counted over
    [0, bins) it is exactly bincount (sliced to bins, cast back).
    """
    graph = exported.graph_module.graph
    replaced = 0
    for node in list(graph.nodes):
        if (node.op == "call_function"
                and str(node.target) == "aten.histc.default"):
            x, bins = node.args[0], node.args[1]
            with graph.inserting_before(node):
                flat = graph.call_function(
                    torch.ops.aten.reshape.default, (x, [-1]))
                cast = graph.call_function(
                    torch.ops.aten._to_copy.default, (flat,),
                    {"dtype": torch.int64})
                counts = graph.call_function(
                    torch.ops.aten.bincount.default, (cast,),
                    {"minlength": bins})
                sliced = graph.call_function(
                    torch.ops.aten.slice.Tensor, (counts, 0, 0, bins))
                out = graph.call_function(
                    torch.ops.aten._to_copy.default, (sliced,),
                    {"dtype": torch.int32})
            node.replace_all_uses_with(out)
            graph.erase_node(node)
            replaced += 1
    exported.graph_module.recompile()
    return replaced


def strip_asserts(module):
    """Remove _assert_tensor_metadata guard nodes.

    They are eager-mode sanity checks Dynamo cannot re-trace when
    compiling on CUDA, and ep.module() regenerates some in submodules —
    strip every graph in the tree.
    """
    removed = 0
    for _, sub in module.named_modules():
        if not hasattr(sub, "graph"):
            continue
        for node in list(sub.graph.nodes):
            if (node.op == "call_function"
                    and "_assert_tensor_metadata" in str(node.target)):
                sub.graph.erase_node(node)
                removed += 1
        sub.recompile()
    return removed


def load_program(graph_path, tensors, device):
    """Load a .pt2, place real tensors into it, retarget to device."""
    exported = torch.export.load(graph_path)

    def place(fqn, tensor):
        if fqn in exported.state_dict:
            exported.state_dict[fqn] = torch.nn.Parameter(
                tensor, requires_grad=False)
        elif fqn in exported.constants:
            exported.constants[fqn] = tensor
        # A graph need not lift every artifact tensor (prefill vs
        # decode differ slightly); extras are simply unused here.

    for fqn, tensor in tensors.items():
        place(fqn, tensor)

    # Example inputs were recorded on meta; materialize placeholders so
    # the device pass can walk them.
    ex_args, ex_kwargs = exported.example_inputs
    fix = lambda t: (torch.zeros(t.shape, dtype=t.dtype)  # noqa: E731
                     if isinstance(t, torch.Tensor) and t.is_meta else t)
    exported._example_inputs = (
        tuple(fix(t) for t in ex_args),
        {k: fix(v) for k, v in ex_kwargs.items()})

    exported = move_to_device_pass(exported, device)
    if device == "cpu":
        cpu_compat(exported)
    return exported


def share_state(src_module, dst_module, state_fqns):
    """Alias state buffers so prefill and decode share one cache.

    move_to_device_pass copies tensors per program; re-point the decode
    module's state buffers at the prefill module's tensors.
    """
    src = dict(src_module.named_buffers())
    shared = 0
    for fqn in state_fqns:
        if fqn in src:
            parent, name = (fqn.rsplit(".", 1)
                            if "." in fqn else ("", fqn))
            dst_parent = (dst_module.get_submodule(parent)
                          if parent else dst_module)
            if name in dst_parent._buffers:
                dst_parent._buffers[name] = src[fqn]
                shared += 1
    return shared
