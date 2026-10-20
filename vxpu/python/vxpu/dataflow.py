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

"""The decode step as a dataflow graph of nodes connected by mailboxes (#372).

A decode step split at its host ops (segments.py) is already a graph:
the dense segments and the host ops (the routed experts, the n-gram
lookup) are its nodes, the tensors one passes to the next are its edges.
Here that graph is made explicit, so that where each node runs, and how
each edge travels, become the executor's choices:

  node       a segment: dense (compiled or eager), experts or engram,
             with the decoder layers it covers
  edge       one output of a node read by another node: shape, dtype,
             bytes per step
  inbox      a buffer at the consumer, allocated once (fixed address):
             the producer writes its output into it, and the consumer
             fires when all of its inboxes are full
  resident   what a node reads that does not travel: the weights and the
             session state (caches), bound where the node runs
  placement  the device each node runs on

The Runner here is the reference: one thread, firing nodes in the order
their inboxes fill, every edge copied into its inbox. It computes what
the split module computes (the same segments, the same arguments), so
the engine can run its decode through it (plan choice dataflow) and the
selftest checks it token for token. The native runner (no Python inside
a step), the experts as nodes of their own, and transports between
devices and machines build on the same graph and inbox plan (#372).
"""

import collections
import operator

import torch

from . import segments


class Ref:
    """An argument of a node: a step input (a placeholder of the split),
    a resident tensor (an attribute of the split), or an edge (output
    `index` of node `node`, index None for a node's only output)."""

    __slots__ = ("kind", "name", "node", "index")

    def __init__(self, kind, name=None, node=None, index=None):
        self.kind, self.name, self.node, self.index = kind, name, node, index

    def __repr__(self):
        if self.kind == "edge":
            return f"{self.node}" + ("" if self.index is None else f"[{self.index}]")
        return f"{self.kind}:{self.name}"


class Node:
    """One segment of the step."""

    def __init__(self, name, kind, layers, args, kwargs):
        self.name = name
        self.kind = kind  # "dense", "experts" or "engram"
        self.layers = layers
        self.args = args  # [Ref]
        self.kwargs = kwargs  # {name: Ref}

    def refs(self):
        return list(self.args) + list(self.kwargs.values())

    def __repr__(self):
        return f"{self.name}({self.kind}, layers {self.layers})"


class Graph:
    """The nodes of a split decode step in call order, its step inputs,
    and the reference of what it returns."""

    def __init__(self, nodes, inputs, output):
        self.nodes = nodes
        self.by_name = {n.name: n for n in nodes}
        self.inputs = inputs
        self.output = output  # Ref, or a tuple of Refs

    @classmethod
    def from_split(cls, split):
        """The graph of a module split at its host ops (split_module's
        output: placeholders, get_attr, call_module per segment, getitem
        to unpack a segment's outputs, output)."""
        from .telemetry import segment_info
        info = {seg["name"]: seg for seg in segment_info(split)}
        refs, nodes, inputs, output = {}, [], [], None

        def ref(arg):
            if isinstance(arg, torch.fx.Node):
                return refs[arg]
            if isinstance(arg, (tuple, list)):
                return type(arg)(ref(a) for a in arg)
            return arg  # a constant argument, passed as it is

        for fx in split.graph.nodes:
            if fx.op == "placeholder":
                inputs.append(fx.target)
                refs[fx] = Ref("input", name=fx.target)
            elif fx.op == "get_attr":
                refs[fx] = Ref("resident", name=fx.target)
            elif fx.op == "call_module":
                seg = info[fx.target]
                nodes.append(Node(fx.target, seg["kind"], seg["layers"],
                                  [ref(a) for a in fx.args], {k: ref(v) for k, v in fx.kwargs.items()}))
                refs[fx] = Ref("edge", node=fx.target)
            elif fx.op == "call_function" and fx.target is operator.getitem:
                base = refs[fx.args[0]]
                if base.kind != "edge" or base.index is not None:
                    raise ValueError(f"dataflow: {fx.name} unpacks something other than a segment's outputs")
                refs[fx] = Ref("edge", node=base.node, index=fx.args[1])
            elif fx.op == "output":
                output = ref(fx.args[0])
            else:
                raise ValueError(f"dataflow: unexpected top-level node {fx.op} {fx.target} in the split")
        return cls(nodes, inputs, output)

    def edges(self):
        """{(producer, index): [consumer names]}, in call order."""
        out = collections.OrderedDict()
        for node in self.nodes:
            for r in _flat(node.refs()):
                if isinstance(r, Ref) and r.kind == "edge":
                    out.setdefault((r.node, r.index), []).append(node.name)
        return out

    def describe(self):
        kinds = collections.Counter(n.kind for n in self.nodes)
        return (f"dataflow: {len(self.nodes)} nodes ("
                + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) + f"), {len(self.edges())} edges")


class Runner(torch.nn.Module):
    """The reference dataflow runner for one split decode step: inboxes
    allocated on the first step (at the consumer's device), then every
    step each node's outputs are written into its consumers' inboxes, and
    a node fires once all of them are full.

    Stands in for segments.Split in the engine (the same split, the same
    recorder hooks and calling convention)."""

    def __init__(self, split, total, compiled, placement=None, device=None):
        super().__init__()
        self.split = split  # the modules and the resident tensors live here
        self.vxpu_segments = (total, compiled)
        self.graph = Graph.from_split(split)
        self.placement = dict(placement or {})  # node name -> device; default: where its resident tensors are (place)
        self.device = torch.device(device) if device is not None else None
        self.recorder = None
        self.inboxes = {}  # {(producer, index, device): tensor}, each allocated the first time it is written
        self._index = {name: i for i, (name, _) in enumerate(segments.segments_of(split))}
        self.place(self.placement)

    def place(self, placement):
        """Run each node of placement ({node name: device}) there, the rest
        where their resident tensors are. The inboxes are allocated again
        by the next step, at the new consumers' devices. A node's resident
        tensors are not moved: placing a node away from its weights is for
        nodes that hold none of their own (the host ops)."""
        self.placement = dict(placement)
        self.inboxes = {}
        self._devices = {n.name: self._device_of(n) for n in self.graph.nodes}
        # per producer: [(index, consumer devices, [(consumer, uses)])]
        uses = collections.Counter()
        for node in self.graph.nodes:
            for r in _flat(node.refs()):
                if isinstance(r, Ref) and r.kind == "edge":
                    uses[(r.node, r.index, node.name)] += 1
        self._waiting = collections.Counter()
        self._fanout = collections.defaultdict(list)
        for (producer, index), consumers in self.graph.edges().items():
            consumers = list(dict.fromkeys(consumers))
            devices = list(dict.fromkeys(self._devices[c] for c in consumers))
            self._fanout[producer].append((index, devices, [(c, uses[(producer, index, c)]) for c in consumers]))
            for c in consumers:
                self._waiting[c] += uses[(producer, index, c)]
        self._returns = {r.node for r in _flat([self.graph.output]) if isinstance(r, Ref) and r.kind == "edge"}

    def set_recorder(self, recorder):
        """Time every node into recorder (telemetry.Recorder), as Split does."""
        self.recorder = recorder

    def _device_of(self, node):
        if node.name in self.placement:
            return torch.device(self.placement[node.name])
        for r in _flat(node.refs()):
            if isinstance(r, Ref) and r.kind == "resident":
                t = _attr(self.split, r.name)
                if isinstance(t, torch.Tensor):
                    return t.device
        return self.device or torch.device("cpu")

    def inbox_plan(self):
        """[{producer, index, device, shape, dtype, bytes}] for every
        inbox allocated so far (all of them after a step)."""
        return [{"producer": p, "index": i, "device": str(d), "shape": list(t.shape), "dtype": str(t.dtype),
                 "bytes": t.numel() * t.element_size()} for (p, i, d), t in self.inboxes.items()]

    def forward(self, *args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()  # as Split: a CUDA-graph replay may reuse its outputs
        recorder = self.recorder
        if recorder is not None:
            recorder.begin_step()
        inputs = dict(zip(self.graph.inputs, args), **kwargs)
        waiting = collections.Counter(self._waiting)
        ready = collections.deque(n.name for n in self.graph.nodes if not waiting[n.name])
        returned = {}
        fired = 0
        while ready:
            node = self.graph.by_name[ready.popleft()]
            dev = self._devices[node.name]
            out = getattr(self.split, node.name)(*[self._arg(r, dev, inputs) for r in node.args],
                                                 **{k: self._arg(r, dev, inputs) for k, r in node.kwargs.items()})
            fired += 1
            if recorder is not None:
                recorder.after_segment(self._index[node.name])
            if node.name in self._returns:
                returned[node.name] = out
            for index, devices, consumers in self._fanout[node.name]:
                value = out if index is None else out[index]
                for d in devices:  # one inbox per consumer device
                    key = (node.name, index, d)
                    if key not in self.inboxes:  # allocated once, then the same address every step
                        self.inboxes[key] = torch.empty_like(value, device=d)
                    self.inboxes[key].copy_(value)
                for consumer, n in consumers:
                    waiting[consumer] -= n
                    if not waiting[consumer]:
                        ready.append(consumer)
        if fired != len(self.graph.nodes):
            raise RuntimeError(f"dataflow: {len(self.graph.nodes) - fired} nodes never had all their inputs")
        if recorder is not None:
            recorder.end_step()
        result = self._resolve(self.graph.output, returned)
        result = result[0] if isinstance(result, (tuple, list)) and len(result) == 1 else result
        return result.clone() if isinstance(result, torch.Tensor) else result

    def _arg(self, r, device, inputs):
        if isinstance(r, (tuple, list)):
            return type(r)(self._arg(x, device, inputs) for x in r)
        if not isinstance(r, Ref):
            return r
        if r.kind == "input":
            return inputs[r.name]
        if r.kind == "resident":
            return _attr(self.split, r.name)
        return self.inboxes[(r.node, r.index, device)]

    def _resolve(self, out, returned):
        if isinstance(out, (tuple, list)):
            return type(out)(self._resolve(o, returned) for o in out)
        if isinstance(out, Ref) and out.kind == "edge":
            value = returned[out.node]
            return value if out.index is None else value[out.index]
        return out


def _flat(refs):
    for r in refs:
        if isinstance(r, (tuple, list)):
            yield from _flat(r)
        else:
            yield r


def _attr(module, target):
    """The attribute a get_attr node reads (a dotted path, read each step:
    the engine may re-point it)."""
    obj = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj
