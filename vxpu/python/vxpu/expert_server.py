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

"""An owner of experts in another process: it answers mailboxes over TCP (#372).

The process holds the expert banks of some layers of an artifact, loaded
as the engine loads them, and answers the engine's decode messages with
the CPU kernels, exactly as the engine's own CPU owner would
(mailboxes.CpuOwner). An engine whose plan names it (expert_remote,
VXPU_EXPERT_REMOTE=host:port) sends it the pairs of those layers that no
GPU pool holds.

    python -m vxpu.expert_server ARTIFACT [--host 127.0.0.1] [--port 50061] [--layers 0,2]

There is no authentication: it listens on the loopback interface unless
told otherwise, and should only be reachable on a private network
(--host 0.0.0.0 to serve other machines).

The wire format, both ways: a 4-byte big-endian header length, a JSON
header, then the raw tensors it describes, in order:

  hello   -> {"op": "hello"}
          <- {"layers": [...], "dim": D, "experts": E, "checkpoint": ID}
  answer  -> {"op": "answer", "layer": L, "n": n, "m": m}
             x bf16 [n, D], tokens int32 [m], experts int32 [m], weights fp32 [m]
          <- {"m": m}, replies bf16 [m, D]
          or {"error": "..."}, and the server closes the connection

ID is checkpoint_id: the same checkpoint has it at any path. Requests are
checked against the server's limits before anything is allocated.

Replies are bf16 on the wire without loss: each is the bf16-rounded down
projection already (mailboxes.py).
"""

import argparse
import hashlib
import json
import os
import socket
import socketserver
import struct

import torch


MAX_HEADER = 64 << 10  # bytes
MAX_ROWS = 4096  # tokens in one request: a prefill chunk, at most


def checkpoint_id(manifest):
    """The checkpoint a manifest's experts come from, independent of where
    it was read from: a digest of its files' sha256s."""
    return hashlib.sha256("\n".join(sorted(manifest["files"])).encode()).hexdigest()[:16]


def send(sock, header, *tensors):
    blob = json.dumps(header).encode()
    parts = [struct.pack(">I", len(blob)), blob]
    parts += [t.contiguous().view(torch.uint8).numpy().tobytes() for t in tensors]
    sock.sendall(b"".join(parts))


def recv_header(sock):
    (size,) = struct.unpack(">I", _recv_exact(sock, 4))
    if size > MAX_HEADER:
        raise ValueError(f"expert server: a {size}-byte header (at most {MAX_HEADER})")
    return json.loads(_recv_exact(sock, size))


def recv_tensor(sock, dtype, shape):
    count = 1
    for s in shape:
        count *= s
    nbytes = count * torch.empty((), dtype=dtype).element_size()
    return torch.frombuffer(bytearray(_recv_exact(sock, nbytes)), dtype=dtype).view(*shape)


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("expert server: connection closed mid-message")
        got += k
    return bytes(buf)


def load(artifact_dir, layers=None):
    """({layer: MoE module}, manifest) for the artifact's external expert
    banks, as the engine places them (plan choices external_dir and
    experts_placement)."""
    from . import deepseek_v41 as ds
    from .engine import external_files_dir, load_expert_layers
    from .plan import Plan
    with open(os.path.join(artifact_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(artifact_dir, "binding.json")) as f:
        binding = json.load(f)
    plan = Plan("cpu")
    if plan["blob_dir"]:  # this machine's local weight tier, as the engine reads it
        from . import localblobs
        manifest["files"], summary = localblobs.localize(manifest["files"], plan["blob_dir"], plan["blob_bucket"])
        print(f"[vxpu] local weight tier: {summary['local']} of {summary['files']} files on local disk", flush=True)
    mm = external_files_dir(plan["external_dir"], manifest)
    args = ds.model_args(manifest["source"]["repo"], int(binding.get("max_cache_len", 1024)))
    experts, _ = load_expert_layers(manifest, (binding.get("external") or {}).get("experts", {}), args, None, mm,
                                    in_ram=plan["experts_placement"] == "ram", layers=layers)
    return experts, manifest


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        from .mailboxes import CpuOwner
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        experts, owner = self.server.experts, CpuOwner()
        some = next(iter(experts.values()))
        while True:
            try:
                header = recv_header(sock)
            except ConnectionError:
                return
            try:
                if header.get("op") == "hello":
                    send(sock, {"layers": sorted(experts), "dim": some.dim, "experts": some.n_local_experts,
                                "checkpoint": self.server.checkpoint})
                    continue
                moe, n, m = self._check(header, experts)
                x = recv_tensor(sock, torch.bfloat16, (n, moe.dim))
                tokens = recv_tensor(sock, torch.int32, (m,)).long()
                chosen = recv_tensor(sock, torch.int32, (m,)).long()
                weights = recv_tensor(sock, torch.float32, (m,))
                if m and (int(tokens.min()) < 0 or int(tokens.max()) >= n):
                    raise ValueError(f"token indices outside [0, {n})")
                replies = owner.answer(moe, x, tokens, chosen, weights)
                send(sock, {"m": m}, replies.to(torch.bfloat16))
            except ConnectionError:
                return
            except Exception as e:  # the client hears why; the stream may be mid-message, so close it
                try:
                    send(sock, {"error": f"{type(e).__name__}: {e}"})
                finally:
                    return

    @staticmethod
    def _check(header, experts):
        """(module, n, m) for a well-formed answer request within limits."""
        if header.get("op") != "answer":
            raise ValueError(f"unknown op {header.get('op')!r}")
        layer, n, m = header.get("layer"), header.get("n"), header.get("m")
        if layer not in experts:
            raise ValueError(f"layer {layer!r} is not held here (holding {sorted(experts)})")
        moe = experts[layer]
        if not (isinstance(n, int) and 1 <= n <= MAX_ROWS):
            raise ValueError(f"n={n!r}: 1 to {MAX_ROWS} rows")
        if not (isinstance(m, int) and 0 <= m <= n * moe.n_local_experts):
            raise ValueError(f"m={m!r}: 0 to n x experts pairs")
        return moe, n, m


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, experts, checkpoint):
        super().__init__(address, _Handler)
        self.experts, self.checkpoint = experts, checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("artifact")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on; 0.0.0.0 for all (no authentication)")
    p.add_argument("--port", type=int, default=50061)
    p.add_argument("--layers", help="comma-separated layers to hold (default: every layer with external experts)")
    args = p.parse_args(argv)
    layers = {int(x) for x in args.layers.split(",")} if args.layers else None
    experts, manifest = load(args.artifact, layers)
    if not experts:
        raise SystemExit("expert server: the artifact places no experts externally (or none of --layers)")
    from . import kernels
    if kernels.load() is None:  # built or loaded here, not on the first request
        raise SystemExit("expert server: the CPU kernels are unavailable")
    server = Server((args.host, args.port), experts, checkpoint_id(manifest))
    print(f"[vxpu] expert server on {args.host}:{server.server_address[1]}: layers {sorted(experts)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
