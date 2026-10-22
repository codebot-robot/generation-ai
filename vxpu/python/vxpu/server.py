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

"""vXPU executor server: receives artifacts, serves inference.

LoadModel is idempotent — the artifact's content digest identifies the
model, so a matching request reuses the loaded engine (sub-second)
instead of reloading. An idle reaper evicts the engine after
--keep-alive seconds so bursts amortize one load but idle accelerators
come back.

Usage:
    python -m vxpu.server --device cuda --compile
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from concurrent import futures

import grpc
import torch

from . import vxpu_pb2, vxpu_pb2_grpc
from .engine import Engine, SamplingParams
from .plan import Plan, probe_hardware

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


class ExecutorServicer(vxpu_pb2_grpc.ExecutorServicer):
    def __init__(self, device="cpu", compile_decode=False,
                 keep_alive_s=3600, work_dir="/tmp/vxpu"):
        self.device = device
        self.compile_decode = compile_decode
        self.work_dir = work_dir
        self.engine = None
        self.digest = None
        self.loading = None  # digest currently loading, if any
        self.load_error = None
        self.last_used = time.time()
        self.keep_alive_s = keep_alive_s
        self.lock = threading.Lock()
        self.experts = None  # ((checkpoint, layers), expert_server.Server) this executor serves (ServeExperts)
        threading.Thread(target=self._reap_idle, daemon=True).start()

    def _touch(self):
        self.last_used = time.time()

    def _reap_idle(self):
        while True:
            time.sleep(5)
            with self.lock:
                idle = time.time() - self.last_used
                if self.engine is not None and idle > self.keep_alive_s:
                    print(f"[vxpu] evicting model after {idle:.0f}s idle",
                          flush=True)
                    self.engine = None
                    self.digest = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _load(self, digest, artifact_dir, total_bytes, plan_given):
        try:
            print(f"[vxpu] loading {digest[:12]} "
                  f"({total_bytes / 1e9:.1f} GB of weights to rehydrate)",
                  flush=True)
            started = time.time()
            engine = Engine(artifact_dir, device=self.device,
                            compile_decode=self.compile_decode,
                            plan_given=plan_given)
            with self.lock:
                if self.loading == digest:
                    self.engine = engine
                    self.digest = digest
                    self.loading = None
                    self._touch()  # idle clock starts after the work
            print(f"[vxpu] ready in {time.time() - started:.0f}s "
                  f"({engine.bytes_fetched / 1e9:.1f} GB downloaded)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] load failed: {e}", file=sys.stderr, flush=True)
            with self.lock:
                if self.loading == digest:
                    self.load_error = str(e)
                    self.loading = None

    def LoadModel(self, request, context):
        """Accepts the artifact and loads asynchronously: clients poll
        NewSession, which reports FAILED_PRECONDITION until ready. No
        long-held RPC, so tunnels (port-forward) never sit idle."""
        try:
            # the coordinator's plan is part of what is loaded: the same
            # artifact under another plan is another engine
            plan_given = dict(request.plan)
            try:  # a plan this executor refuses fails the call, not the load behind it
                Plan(self.device, compile_decode=self.compile_decode, given=plan_given)
            except ValueError as e:  # ConfigError, and the plain ValueErrors of a switch's own checks
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"plan refused: {e}")
                return vxpu_pb2.LoadModelResponse(success=False)
            # the files are the artifact's alone; the engine is the artifact under its plan
            artifact_digest = hashlib.sha256(
                request.manifest_json.encode()
                + request.binding_json.encode()
                + request.prefill_graph
                + request.decode_graph).hexdigest()
            digest = hashlib.sha256((artifact_digest + json.dumps(plan_given, sort_keys=True)).encode()).hexdigest()
            with self.lock:
                self._touch()
                if self.engine is not None and digest == self.digest:
                    print(f"[vxpu] {digest[:12]} already loaded; reusing",
                          flush=True)
                    return vxpu_pb2.LoadModelResponse(success=True)
                if self.loading == digest:
                    return vxpu_pb2.LoadModelResponse(success=True)

                # Reset currently resident engine so NewSession will wait for this load.
                self.engine = None
                self.digest = None
                self.loading = digest
                self.load_error = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                artifact_dir = os.path.join(self.work_dir, artifact_digest[:16])
                os.makedirs(artifact_dir, exist_ok=True)
                for name, data in (
                        ("manifest.json", request.manifest_json.encode()),
                        ("binding.json", request.binding_json.encode()),
                        ("prefill.pt2", request.prefill_graph),
                        ("decode.pt2", request.decode_graph)):
                    with open(os.path.join(artifact_dir, name), "wb") as f:
                        f.write(data)

                manifest = json.loads(request.manifest_json)
                total = sum(f["size"]
                            for f in manifest["files"].values())
                threading.Thread(
                    target=self._load,
                    args=(digest, artifact_dir, total, plan_given),
                    daemon=True).start()
                return vxpu_pb2.LoadModelResponse(success=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] LoadModel error: {e}", file=sys.stderr,
                  flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.LoadModelResponse(success=False)

    def ServeExperts(self, request, context):
        """Serve some layers' experts of an artifact to other executors'
        mailboxes (expert_server.py), on a port of this machine: how the
        coordinator spreads a model's experts over executors (#394). The
        banks are loaded before it answers. One set at a time: a new one
        replaces the old. Like the expert server, no authentication: the
        executors' network is private."""
        from . import expert_server, kernels
        try:
            manifest = json.loads(request.manifest_json)
            layers = sorted(set(request.layers))
            key = (expert_server.checkpoint_id(manifest), tuple(layers))
            with self.lock:
                if self.experts is not None and self.experts[0] == key:
                    return vxpu_pb2.ServeExpertsResponse(port=self.experts[1].server_address[1])
            artifact_dir = os.path.join(self.work_dir, "experts-" + key[0][:16])
            os.makedirs(artifact_dir, exist_ok=True)
            for name, data in (("manifest.json", request.manifest_json), ("binding.json", request.binding_json)):
                with open(os.path.join(artifact_dir, name), "w") as f:
                    f.write(data)
            experts, _ = expert_server.load(artifact_dir, set(layers))
            missing = sorted(set(layers) - set(experts))
            if not layers or missing:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"the artifact places no experts externally for layers {missing or layers}")
                return vxpu_pb2.ServeExpertsResponse()
            if kernels.load() is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("the CPU kernels are unavailable")
                return vxpu_pb2.ServeExpertsResponse()
            server = expert_server.Server(("0.0.0.0", 0), experts, key[0])
            threading.Thread(target=server.serve_forever, name="vxpu-expert-server", daemon=True).start()
            with self.lock:
                old, self.experts = self.experts, (key, server)
            if old is not None:
                old[1].shutdown()
                old[1].server_close()
            print(f"[vxpu] serving experts of layers {layers} on port {server.server_address[1]}", flush=True)
            return vxpu_pb2.ServeExpertsResponse(port=server.server_address[1])
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] ServeExperts error: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.ServeExpertsResponse()

    def Describe(self, request, context):
        """What this executor runs on, for the coordinator's machine
        database (#394): the same probe its plans record."""
        return vxpu_pb2.DescribeResponse(hardware_json=json.dumps(probe_hardware(self.device)))

    def NewSession(self, request, context):
        with self.lock:
            if self.loading:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("model still loading")
                return vxpu_pb2.NewSessionResponse()
            if self.load_error:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"model load failed: {self.load_error}")
                return vxpu_pb2.NewSessionResponse()
            if self.engine is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("call LoadModel first")
                return vxpu_pb2.NewSessionResponse()
            try:
                self._touch()
                session_id = self.engine.new_session()
            except Exception as e:  # noqa: BLE001
                print(f"[vxpu] NewSession error: {e}", file=sys.stderr,
                      flush=True)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return vxpu_pb2.NewSessionResponse()

        print(f"[vxpu] new session {session_id}", flush=True)
        return vxpu_pb2.NewSessionResponse(session_id=session_id)

    def Chat(self, request, context):
        with self.lock:
            if self.engine is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("call LoadModel first")
                return vxpu_pb2.ChatResponse()
            self._touch()
            engine = self.engine

        if request.max_new_tokens <= 0:
            max_new_tokens = 96
        elif request.max_new_tokens > 4096:
            max_new_tokens = 4096
        else:
            max_new_tokens = request.max_new_tokens

        try:
            reply = engine.chat(
                request.session_id, request.text,
                max_new_tokens=max_new_tokens,
                raw_prompt=request.raw_prompt,
                telemetry=request.telemetry)
            return vxpu_pb2.ChatResponse(
                text=reply["text"],
                session_tokens=reply["session_tokens"],
                new_prompt_tokens=reply["new_prompt_tokens"],
                generated=reply["generated"],
                prefill_ms=float(reply["prefill_ms"]),
                ms_per_token=float(reply["ms_per_token"]),
                telemetry_json=json.dumps(reply["telemetry"]) if "telemetry" in reply else "")
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return vxpu_pb2.ChatResponse()
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] Chat error: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.ChatResponse()


    def Generate(self, request, context):
        with self.lock:
            if self.engine is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              "call LoadModel first")
            self._touch()
            engine = self.engine

        if request.HasField("max_new_tokens"):
            max_new_tokens = max(0, request.max_new_tokens)
        else:
            max_new_tokens = 20  # transformers' max_length default
        max_new_tokens = min(max_new_tokens, 4096)
        params = SamplingParams(
            do_sample=(request.do_sample
                       if request.HasField("do_sample") else None),
            temperature=(request.temperature
                         if request.HasField("temperature") else None),
            top_k=request.top_k if request.HasField("top_k") else None,
            top_p=request.top_p if request.HasField("top_p") else None,
            repetition_penalty=(request.repetition_penalty
                                if request.HasField("repetition_penalty")
                                else None),
            seed=request.seed if request.HasField("seed") else None)
        try:
            for chunk in engine.generate(
                    request.session_id, list(request.input_ids),
                    max_new_tokens, params, list(request.eos_token_id),
                    telemetry=request.HasField("telemetry") and request.telemetry):
                if not context.is_active():
                    return
                self._touch()  # a long stream is not idle
                if chunk["done"]:
                    yield vxpu_pb2.GenerateResponse(
                        done=True,
                        finish_reason=chunk["finish_reason"],
                        prompt_tokens=chunk["prompt_tokens"],
                        prefilled_tokens=chunk["prefilled_tokens"],
                        generated=chunk["generated"],
                        prefill_ms=float(chunk["prefill_ms"]),
                        ms_per_token=float(chunk["ms_per_token"]),
                        telemetry_json=(json.dumps(chunk["telemetry"])
                                        if chunk.get("telemetry") else ""))
                else:
                    yield vxpu_pb2.GenerateResponse(
                        token_ids=chunk["token_ids"])
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] Generate error: {e}", file=sys.stderr, flush=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compile", action="store_true",
        default=torch.cuda.is_available(),
        help="torch.compile the decode graph per session")
    parser.add_argument("--keep-alive", type=int, default=3600,
                        help="seconds to keep an idle model loaded")
    args = parser.parse_args()

    print(f"[vxpu] executor on :{args.port} (device={args.device}, "
          f"compile={args.compile}, keep-alive={args.keep_alive}s, "
          f"threads={torch.get_num_threads()})",
          flush=True)
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        # Clients ping to keep port-forward tunnels alive through
        # long LoadModel calls; accept them.
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10), options=options)
    vxpu_pb2_grpc.add_ExecutorServicer_to_server(
        ExecutorServicer(device=args.device,
                         compile_decode=args.compile,
                         keep_alive_s=args.keep_alive), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
