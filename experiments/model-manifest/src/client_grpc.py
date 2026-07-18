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

"""gRPC client for testing model-router.

Loads model-manifest files locally, registers them with the router to get a
prepared model ID, then runs concurrent multi-turn chat sessions through the router.
"""

import argparse
import os
import sys

import grpc

import router_pb2
import router_pb2_grpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="localhost:50051", help="router endpoint")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="model identification key")
    parser.add_argument("--manifest", default="HuggingFaceTB--SmolLM2-135M-Instruct.manifest.json")
    parser.add_argument("--binding", default="cached.binding.json")
    parser.add_argument("--prefill", default="prefill.pt2")
    parser.add_argument("--decode", default="decode.pt2")
    args = parser.parse_args()

    print("[client] Loading manifest and graph files...", flush=True)
    if not os.path.exists(args.manifest):
        print(f"Error: manifest file {args.manifest} not found. Run manifest.py and export_graph.py first.")
        sys.exit(1)

    with open(args.manifest, "r") as f:
        manifest_json = f.read()
    with open(args.binding, "r") as f:
        binding_json = f.read()
    with open(args.prefill, "rb") as f:
        prefill_graph = f.read()
    with open(args.decode, "rb") as f:
        decode_graph = f.read()

    print(f"[client] Connecting to model-router at {args.router}...", flush=True)
    max_msg_size = 128 * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", max_msg_size),
        ("grpc.max_send_message_length", max_msg_size)
    ]
    with grpc.insecure_channel(args.router, options=options) as channel:
        router_stub = router_pb2_grpc.ModelRouterStub(channel)

        print(f"[client] Preparing model {args.model_id} with router...", flush=True)
        prep_req = router_pb2.PrepareModelRequest(
            model_id=args.model_id,
            manifest_json=manifest_json,
            binding_json=binding_json,
            prefill_graph=prefill_graph,
            decode_graph=decode_graph
        )
        try:
            prep_resp = router_stub.PrepareModel(prep_req)
            prepared_model_id = prep_resp.prepared_model_id
            print(f"[client] Router prepared model ID: {prepared_model_id}", flush=True)
        except grpc.RpcError as e:
            print(f"[client] Router PrepareModel gRPC Error: {e.code()} - {e.details()}", file=sys.stderr)
            sys.exit(1)

        print("[client] Creating sessions through router...", flush=True)
        sess_a = router_stub.NewSession(router_pb2.NewSessionRequest(model_id=prepared_model_id)).session_id
        sess_b = router_stub.NewSession(router_pb2.NewSessionRequest(model_id=prepared_model_id)).session_id
        print(f"[client] Created routed sessions: {sess_a}, {sess_b}", flush=True)

        def turn(session, text):
            print(f"[{session}] >>> {text}", flush=True)
            chat_req = router_pb2.ChatRequest(
                session_id=session,
                text=text,
                max_new_tokens=96
            )
            reply = router_stub.Chat(chat_req)
            print(f"[{session}] {reply.text}", flush=True)
            print(f"[{session}]     ({reply.generated} tokens, "
                  f"{reply.ms_per_token:.1f} ms/token, prefill "
                  f"{reply.new_prompt_tokens} new tokens in "
                  f"{reply.prefill_ms:.1f} ms, session total "
                  f"{reply.session_tokens})\n", flush=True)

        turn(sess_a, "Why is the sky blue?")
        turn(sess_b, "What is 2 + 2?")
        turn(sess_a, "Summarize your previous answer in one short sentence.")
        turn(sess_b, "What number did I just ask you about?")


if __name__ == "__main__":
    main()
