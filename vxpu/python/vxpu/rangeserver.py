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

"""Serving a range of whole layers to the engine that runs the layers before it, over tcp:// (#387).

An engine started with VXPU_LAYER_RANGE runs only some of the model's layers.
A RangeServer (csrc/node_server.cpp) lets the engine before it send each step
over the network. Messages use the node wire format, and each one starts with
the head engine's session id, as bytes. The node number says what the
message is:

  node 0  a decode step. The inputs come in range_inputs order. The reply is
          the range's outputs: the step's logits, if this is the last range.
  node 1  a prefill, in the same form.
  node 2  reset the session's state (the head started the session over).
  node 3  describe the range: the names of its inputs for each phase, as JSON.

Decode steps start out in Python. Once the range's step graph is captured (on
its second decode step), the server runs each decode step for the current
session in C++: it copies the inputs into the graph's buffers, starts the
step host, replays the graphs and copies the outputs back. Prefills, resets
and steps for other sessions still go through Python.

RemoteRange is the head engine's side (Engine.attach_ranges).
"""

import json

import torch

from . import kernels


def _session_tensor(session_id):
    return torch.tensor(list(session_id.encode()), dtype=torch.uint8)


def _session_of(t):
    return bytes(t.tolist()).decode()


def serve(engine, port=0):
    """Serve the engine's range of layers on a port (0 picks a free one).
    Returns the RangeServer, which has port(), stop() and native_steps()."""
    k = kernels.load()
    if k is None:
        raise RuntimeError(f"a range server needs the CPU kernels: {kernels.why_not()}")
    names = {phase: engine.range_inputs(phase) for phase in ("decode", "prefill")}
    numbers = {phase: engine.range_numbers(phase) for phase in ("decode", "prefill")}
    holder = {}

    def outputs(out):
        if isinstance(out, dict):
            return [out[n] for n in sorted(out)]
        return [out]

    def fallback(node, tensors):
        if node == 3:  # describe: inputs per phase, and for a middle range the outputs the next range needs (sorted by name)
            graph = getattr(engine._decode_run, "runner", engine._decode_run).graph
            doc = {"inputs": names, "outputs": sorted(graph.output_names) if graph.output_names is not None else None}
            return [torch.tensor(list(json.dumps(doc).encode()), dtype=torch.uint8)]
        session = _session_of(tensors[0])
        if node == 2:
            engine.reset_range(session)
            return []
        phase = "decode" if node == 0 else "prefill"
        values = {n: t.item() if n in numbers[phase] else t for n, t in zip(names[phase], tensors[1:])}
        out = engine.run_range(session, phase, values)
        if phase == "decode":
            _arm_native(engine, holder.get("server"))
        return [t.cpu() if isinstance(t, torch.Tensor) else torch.tensor(t) for t in outputs(out)]

    server = k.RangeServer(port, fallback)
    holder["server"] = server
    print(f"[vxpu] range server for layers {engine._range_text()} on port {server.port()}", flush=True)
    return server


def _arm_native(engine, server):
    """Hand the range's captured decode step graph to the server, once.
    After that, decode steps run in C++ (node_server.cpp)."""
    from .stepgraph import StepGraph
    sg = engine._decode_split
    if server is None or not isinstance(sg, StepGraph) or sg.graph is None or getattr(sg, "_served", False):
        return
    out = sg._output
    outs = [out[n] for n in sorted(out)] if isinstance(out, dict) else (list(out) if isinstance(out, (tuple, list)) else [out])
    if sg.runner.graph.output_names is not None:  # a middle range: outputs sorted by name, as the Python fallback sends them
        outs = [o for _, o in sorted(zip(sg.runner.graph.output_names, out))]
    dev = engine.device if isinstance(engine.device, torch.device) else torch.device(engine.device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    execs = [sg.graph.raw_cuda_graph_exec()] + [g.raw_cuda_graph_exec() for _, g in sg._peer_graphs]
    streams = [torch.cuda.current_stream(index).cuda_stream] + [torch.cuda.current_stream(p).cuda_stream
                                                               for p, _ in sg._peer_graphs]
    devices = [index] + [torch.device(p).index for p, _ in sg._peer_graphs]
    server.set_decode(list(sg._inputs), outs, execs, streams, devices, sg._host, sg._flags)
    sg._served = True
    print(f"[vxpu] range server: decode steps replay natively from now on", flush=True)


class RemoteRange:
    """A range of layers served by another engine (see serve), as the head
    engine uses it through Engine.attach_ranges: it reports its inputs for
    each phase, runs a step for a session, and resets a session."""

    def __init__(self, address, device):
        host, port = address.removeprefix("tcp://").rsplit(":", 1)
        self.address, self.device = address, torch.device(device)
        k = kernels.load()
        if k is None:
            raise RuntimeError(f"a remote range needs the CPU kernels: {kernels.why_not()}")
        self.client = k.NodeClient(host, int(port))
        doc = json.loads(bytes(self.client.call(3, [])[0].tolist()).decode())
        self.names, self.outputs = doc["inputs"], doc["outputs"]

    def inputs(self, phase):
        return self.names[phase]

    def run(self, session_id, phase, values):
        tensors = [_session_tensor(session_id)] + [values[n].cpu() if isinstance(values[n], torch.Tensor)
                                                   else torch.tensor(values[n]) for n in self.names[phase]]
        out = self.client.call(0 if phase == "decode" else 1, tensors)
        if self.outputs is None:  # the last range: the step's output
            return out[0].to(self.device)
        return {n: o.to(self.device) for n, o in zip(self.outputs, out)}

    def reset(self, session_id):
        self.client.call(2, [_session_tensor(session_id)])
