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

"""Export entry point for DeepSeek-V4.1-Flash (model_type ``deepseek_v41``).

There is no transformers implementation of this model; the exporter
captures the model maker's own reference module, carried as a fork in
``third_party/deepseek_v41`` with plain-aten versions of its kernels and
static cache state (see that repository's vxpu branch). The flow is the
transformers one: build on the meta device, harvest the constant
buffers, export prefill and decode, put the constants back, and bind
every parameter to the checkpoint by its own name. Routed experts are
``stacked`` banks assembled from the per-expert checkpoint tensors.

The model's arguments come from ``model_args.json`` next to a local
checkpoint when present (test fixtures), else from the fork's
``inference/config.json``, which is the released model's.
"""

import contextlib
import json
import os
import sys

import torch

FORK = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "deepseek_v41", "inference")
MODEL_TYPE = "deepseek_v41"


def _fork():
    """Import the fork's modules (they import each other by bare name)."""
    path = os.path.abspath(os.environ.get("VXPU_DEEPSEEK_V41", FORK))
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("DEEPSEEK_KERNELS", "aten")
    import model as ds_model  # noqa: F401
    return ds_model


@contextlib.contextmanager
def bf16_default():
    """The reference runner sets torch's default dtype to bf16 for the whole
    run and the kernels size their outputs from it; construction, eager
    runs and export must happen under the same setting."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def model_args(repo_or_dir, max_cache_len, model_type_args=None):
    """ModelArgs for the checkpoint: ``model_args.json`` beside a local
    checkpoint, else the fork's released config."""
    ds = _fork()
    local = os.path.join(repo_or_dir, "model_args.json") if os.path.isdir(repo_or_dir) else None
    path = local if local and os.path.exists(local) else os.path.join(os.path.dirname(ds.__file__), "config.json")
    with open(path) as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if k in ds.ModelArgs.__dataclass_fields__}
    raw.update(max_batch_size=1, max_seq_len=max_cache_len, temperature=0)
    return ds.ModelArgs(**raw)


class Prefill(torch.nn.Module):
    """input_ids [1, L], cache_position [L] (always 0..L-1: this model's prefill starts at 0)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache_position):
        return self.model(input_ids, 0)[1].unsqueeze(1)  # [1, 1, vocab]: the last position's logits


class Decode(torch.nn.Module):
    """input_ids [1, 1], cache_position [1]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache_position):
        return self.model(input_ids, cache_position[0])[1].unsqueeze(1)


def build(repo_or_dir, tokenizer, max_cache_len):
    """The model on the meta device with its constant buffers harvested.
    Returns (wrapper-less model, {buffer name: value})."""
    ds = _fork()
    args = model_args(repo_or_dir, max_cache_len)
    with torch.device("meta"), bf16_default():
        model = ds.Transformer(args, tokenizer).eval()
    # remove_duplicate=False: layers share one rotary table object (the fork
    # caches it), and every name needs its value
    constants = {name: buf.detach().clone()
                 for name, buf in model.named_buffers(remove_duplicate=False) if not buf.is_meta}
    model.to("meta")
    return model, constants, args


def set_routing(model, routing):
    for layer in model.layers:
        layer.ffn.routing = routing


def checkpoint_view(model, manifest_tensors):
    """(direct, merged) as export.checkpoint_view returns them, for this
    model: every parameter is named as in the checkpoint, except the
    expert banks, which stack the per-expert tensors."""
    import re
    direct, merged = {}, {}
    names = set(manifest_tensors)
    for name in names:
        direct[name] = name
    bank = re.compile(r"^(?P<prefix>(?:layers|mtp)\.\d+\.ffn)\.experts\.(?P<i>\d+)\.(?P<proj>w[123])\.(?P<kind>weight|scale)$")
    groups = {}
    for name in names:
        m = bank.match(name)
        if m:
            key = f"{m['prefix']}.{m['proj']}" + ("" if m["kind"] == "weight" else "_scale")
            groups.setdefault(key, []).append((int(m["i"]), name))
            direct.pop(name, None)
    for key, parts in groups.items():
        merged[key] = {"parts": [[n] for _, n in sorted(parts)], "cat_dim": 0, "stack": True}
    return direct, merged


def export_graphs(model, max_cache_len, min_prefill_len):
    """Prefill (symbolic length) and decode programs."""
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True
    seq = torch.export.Dim("seq", min=min_prefill_len, max=max_cache_len)
    with bf16_default():
        set_routing(model, "grouped")
        # keyword arguments, as the engine calls the graphs
        prefill = torch.export.export(
            Prefill(model), (),
            kwargs={"input_ids": torch.zeros(1, 8, dtype=torch.long, device="meta"),
                    "cache_position": torch.arange(8, device="meta")},
            dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
            strict=False)
        set_routing(model, "batched")
        decode = torch.export.export(
            Decode(model), (),
            kwargs={"input_ids": torch.zeros(1, 1, dtype=torch.long, device="meta"),
                    "cache_position": torch.zeros(1, dtype=torch.long, device="meta")},
            strict=False)
    return prefill, decode


def binding_extras(args):
    """What the engine must know beyond the tensor classes."""
    return {
        "model_type": MODEL_TYPE,
        # The prefill graph starts at position 0: a changed prefix means a
        # fresh session, not a suffix prefill.
        "prefill_from_zero": True,
        "quantization": {"quant_method": "fp8", "block_size": [32, 32],
                         "expert_dtype": args.expert_dtype or "fp8", "scale_fmt": "ue8m0"},
    }
