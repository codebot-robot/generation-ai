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

"""Serving engine: shared weights, per-session KV caches.

The binding's three categories map directly onto serving:

    bound   -> fetched once, shared read-only across sessions
    derived -> computed once from config, shared
    state   -> allocated fresh per session; the graphs mutate it in
               place, so a session IS its state tensors

Multi-turn conversation costs only the new tokens: the prefill graph
has a dynamic sequence dimension and explicit cache positions, so a
follow-up turn prefills the suffix at positions [cached_len, ...).
"""

import json
import os
import time

import torch

from .rehydrate import (derived_tensor, fetch_tensor, load_program,
                        share_state, strip_asserts)


class Engine:
    def __init__(self, artifact_dir, device="cpu", compile_decode=False,
                 cas_dir=None):
        from transformers import AutoTokenizer
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING)

        self.artifact_dir = artifact_dir
        self.device = device
        self.compile_decode = compile_decode
        with open(os.path.join(artifact_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        with open(os.path.join(artifact_dir, "binding.json")) as f:
            self.binding = json.load(f)
        if cas_dir:
            os.makedirs(cas_dir, exist_ok=True)

        self.config = CONFIG_MAPPING[
            self.manifest["config"]["model_type"]
        ].from_dict(self.manifest["config"])
        # Tokenizer/processor files are not yet part of the manifest.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.manifest["source"]["repo"])

        # Shared, read-only: fetched/computed exactly once. Shapes for
        # derived/state tensors come from the decode program itself.
        probe = torch.export.load(
            os.path.join(artifact_dir, "decode.pt2"))
        probe_meta = {**probe.state_dict, **probe.constants}

        # Weights move to the device once, here: both programs then
        # alias the same storage (device pass no-ops on tensors already
        # in place), so a session costs cache memory, not a weight copy.
        self.shared = {}
        self.bytes_fetched = 0
        for fqn, ref in self.binding["bound"].items():
            tensor, downloaded = fetch_tensor(
                ref, self.manifest["files"], cas_dir)
            self.shared[fqn] = tensor.to(device)
            self.bytes_fetched += downloaded
        for fqn in self.binding["derived"]:
            self.shared[fqn] = derived_tensor(
                fqn, probe_meta[fqn], self.config).to(device)
        self.state_specs = {
            fqn: (probe_meta[fqn].shape, probe_meta[fqn].dtype)
            for fqn in self.binding["state"]}

        self.sessions = {}
        self._next_id = 0

    def new_session(self):
        session_id = f"s{self._next_id}"
        self._next_id += 1

        tensors = dict(self.shared)
        for fqn, (shape, dtype) in self.state_specs.items():
            tensors[fqn] = torch.zeros(shape, dtype=dtype,
                                       device=self.device)

        started = time.perf_counter()
        prefill = load_program(
            os.path.join(self.artifact_dir, "prefill.pt2"),
            tensors, self.device).module()
        decode = load_program(
            os.path.join(self.artifact_dir, "decode.pt2"),
            tensors, self.device).module()
        shared = share_state(prefill, decode, self.binding["state"])
        assert shared > 0, "prefill/decode share no cache state"
        if self.compile_decode:
            strip_asserts(decode)
            decode = torch.compile(decode)

        self.sessions[session_id] = {
            "prefill": prefill,
            "decode": decode,
            "messages": [],
            "cached_len": 0,
            "build_s": time.perf_counter() - started,
        }
        return session_id

    def chat(self, session_id, text, max_new_tokens=96):
        session = self.sessions[session_id]
        session["messages"].append({"role": "user", "content": text})
        ids = self.tokenizer.apply_chat_template(
            session["messages"], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)["input_ids"]
        total_len = ids.shape[1]
        start = session["cached_len"]
        new_ids = ids[:, start:].to(self.device)

        prefill_started = time.perf_counter()
        logits = session["prefill"](
            input_ids=new_ids,
            cache_position=torch.arange(start, total_len,
                                        device=self.device))
        prefill_s = time.perf_counter() - prefill_started

        token_ids, position = [], total_len
        next_id = int(logits[0, -1].argmax())
        loop_started = time.perf_counter()
        for _ in range(max_new_tokens):
            if next_id == self.tokenizer.eos_token_id:
                break
            token_ids.append(next_id)
            logits = session["decode"](
                input_ids=torch.tensor([[next_id]], device=self.device),
                cache_position=torch.tensor([position],
                                            device=self.device))
            next_id = int(logits[0, -1].argmax())
            position += 1
        loop_s = time.perf_counter() - loop_started

        reply = self.tokenizer.decode(token_ids,
                                      skip_special_tokens=True)
        session["messages"].append(
            {"role": "assistant", "content": reply})
        session["cached_len"] = position

        return {
            "text": reply,
            "session_tokens": position,
            "new_prompt_tokens": total_len - start,
            "generated": len(token_ids),
            "prefill_ms": round(prefill_s * 1e3),
            "ms_per_token": round(
                loop_s / max(len(token_ids), 1) * 1e3, 1),
        }
