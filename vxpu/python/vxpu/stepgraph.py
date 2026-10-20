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

"""The whole decode step in one CUDA graph, host ops through mailboxes (#372).

With the split decode run through the dataflow runner (dataflow.py) and
its dense segments as AOTInductor packages (aoti.py, segment_runtime
aoti), every dense node is native code that a CUDA graph can capture.
What still breaks a step into pieces is the host ops: today Python waits
for the GPU before each, runs it, and launches the next segment after,
and the GPU waits for Python in between (on DeepSeek-V4.1-Flash, 43 dense
segments take 20.7 ms a step in the engine and 11.3 ms replayed back to
back, #277).

Here a host op is a mailbox inside the graph instead:

  GPU   copy the node's inputs into pinned host inboxes, then write 1 to
        the node's to-CPU flag (cuStreamWriteValue32, a system-wide fence
        first), then wait for its from-CPU flag to be 1
        (cuStreamWaitValue32), clear it, and copy the reply from the
        pinned outbox into the consumers' inboxes on the GPU
  CPU   a worker thread waits for each host node's to-CPU flag in turn,
        clears it, runs the node on the pinned inputs, writes the reply
        into the pinned outbox, then sets the from-CPU flag

So a decode step is one graph replay: the GPU runs the dense segments
and blocks on the host ops' flags itself; Python launches the step and
waits for its end. The flags and buffers have fixed addresses and fixed
values, so the same graph replays every step.

Hot experts (a placement's GPU pools, experts_gpu.py) take part too. An
experts node whose layer has a pool on the engine's GPU runs the pool's
kernels inside the graph, on a forked stream, while the CPU worker
answers the pairs no GPU holds; a pool on another GPU runs in that GPU's
own graph, replayed alongside, with nothing in host memory: the engine's
GPU copies the node's inputs straight into an inbox on the other GPU and
writes a flag in that GPU's memory; the other GPU, waiting on the flag,
computes, copies its per-pair replies straight back and writes a flag in
the engine's GPU's memory. The engine's graph then takes each
pair's reply from whoever holds the expert and sums them in k order, as
the mailboxes do (mailboxes.py).

The first decode step after a load runs through the dataflow runner
(loading the packages, allocating the inboxes); the second captures the
step and replays it; later steps replay. Telemetry still records the host
ops (the worker runs them in the engine's scope) and the steps; the
segments' own times, which only a node-by-node run can take, are left
out.
"""

import ctypes
import gc
import threading
import time
import weakref

import torch

from . import dataflow, triton_ops

CU_STREAM_WAIT_VALUE_EQ = 0x1
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0  # with a system-wide memory fence before the write


class _Driver:
    """The CUDA driver's stream memory operations, through ctypes."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.write = self.lib.cuStreamWriteValue32_v2
        self.wait = self.lib.cuStreamWaitValue32_v2
        for fn in (self.write, self.wait):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            fn.restype = ctypes.c_int
        self.copy = self.lib.cuMemcpyAsync
        self.copy.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        self.copy.restype = ctypes.c_int
        self.dptr = self.lib.cuMemHostGetDevicePointer_v2
        self.dptr.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        self.dptr.restype = ctypes.c_int

    def device_address(self, pinned):
        out = ctypes.c_uint64()
        _check(self.dptr(ctypes.byref(out), ctypes.c_void_p(pinned.data_ptr()), 0), "cuMemHostGetDevicePointer")
        return out.value

    def write_value(self, stream, address, value):
        _check(self.write(ctypes.c_void_p(stream.cuda_stream), address, value, CU_STREAM_WRITE_VALUE_DEFAULT),
               "cuStreamWriteValue32")

    def memcpy(self, stream, dst, src):
        """dst <- src on stream, whichever devices they are on (unified
        addressing; peer access enabled): a copy node in a capture."""
        _check(self.copy(dst.data_ptr(), src.data_ptr(), src.numel() * src.element_size(),
                         ctypes.c_void_p(stream.cuda_stream)), "cuMemcpyAsync")

    def wait_value(self, stream, address, value):
        _check(self.wait(ctypes.c_void_p(stream.cuda_stream), address, value, CU_STREAM_WAIT_VALUE_EQ),
               "cuStreamWaitValue32")


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"step graph: {what} failed (CUresult {status})")


def _experts_call(sub):
    """For a host segment that calls the routed experts: (layer, the
    positions of x, weights and indices among the segment's inputs)."""
    placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
    for n in sub.graph.nodes:
        if n.op == "call_function" and "moe_experts" in str(getattr(n.target, "__name__", n.target)):
            x, w, i, layer = n.args[:4]
            return int(layer), [placeholders.index(a) for a in (x, w, i)]
    return None, None


class StepGraph(torch.nn.Module):
    """A dataflow runner's step as one CUDA graph (module docstring)."""

    def __init__(self, runner):
        super().__init__()
        self.runner = runner  # dataflow.Runner over a split whose dense segments are aoti.GraphedSegment
        self.split = runner.split
        self.vxpu_segments = runner.vxpu_segments
        self.graph = None
        self._peer_graphs = []
        self.steps = 0
        self.recorder = None
        self._worker = None

    def set_recorder(self, recorder):
        self.recorder = recorder
        self.runner.set_recorder(recorder)

    def forward(self, *args, **kwargs):
        self.steps += 1
        if self.steps == 1:
            return self.runner(*args, **kwargs)  # the first step loads the packages and allocates the inboxes
        if self.graph is None:
            gc.disable()  # a collection inside a capture can unload a kernel's module, which a capture forbids
            try:
                self._capture(args, kwargs)
            finally:
                gc.enable()
        for buf, a in zip(self._inputs, self._input_values(args, kwargs)):
            buf.copy_(a)
        self._worker.go.set()
        self.graph.replay()
        # the other GPUs' graphs after this one's: launched first, a graph
        # that opens by waiting on a flag held the launching thread ~5 ms
        for pdev, g in self._peer_graphs:
            with torch.cuda.device(pdev):
                g.replay()
        torch.cuda.current_stream().synchronize()
        for pdev, _ in self._peer_graphs:
            torch.cuda.current_stream(pdev).synchronize()
        error = self._worker.error
        if error is not None:
            # the worker raised every from-CPU flag to release the GPU, and
            # the to-CPU flags after the failed node are still up: the next
            # step starts clean (the GPU is idle and the worker done)
            self._worker.error = None
            self._worker.view[:] = 0
            raise error
        out = self._output
        out = out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out
        return out.clone() if isinstance(out, torch.Tensor) else out

    def _input_values(self, args, kwargs):
        values = dict(zip(self.runner.graph.inputs, args), **kwargs)
        return [values[name] for name in self.runner.graph.inputs]

    def _capture(self, args, kwargs):
        from . import aoti, ops
        graph, runner = self.runner.graph, self.runner
        driver = _Driver()
        self._inputs = [v.clone() for v in self._input_values(args, kwargs)]
        inputs = dict(zip(graph.inputs, self._inputs))
        host_nodes = [n for n in graph.nodes if n.kind != "dense"]
        produced = {}  # node -> its output tensors on the GPU, as captured
        dev = runner.device
        if dev.index is None:  # the pools name their GPU by index
            dev = torch.device(dev.type, torch.cuda.current_device())
        external = ops._active.external

        def resolve(r):
            if isinstance(r, (tuple, list)):
                return type(r)(resolve(x) for x in r)
            if not isinstance(r, dataflow.Ref):
                return r
            if r.kind == "input":
                return inputs[r.name]
            if r.kind == "resident":
                return dataflow._attr(self.split, r.name)
            out = produced[r.node]
            return out if r.index is None else out[r.index]

        # Every pinned buffer before the captures (allocating pinned memory
        # while a stream captures is not allowed), shaped like the inboxes
        # the runner's first step allocated. An experts node whose layer
        # has pools gets per-pair outboxes: one the CPU fills, one per
        # other GPU holding some of its experts.
        boxes, pooled, peers = {}, {}, []
        for node in host_nodes:
            dev_ = runner._devices[node.name]
            example = [runner._arg(r, dev_, inputs) for r in node.args]
            like = next(t for (p, _, _), t in runner.inboxes.items() if p == node.name)
            out_like, peer_boxes = like, {}
            if node.kind == "experts":
                layer, where = _experts_call(getattr(self.split, node.name))
                res = getattr(external.experts.get(layer), "_vxpu_hot", None) if layer is not None else None
                if res is not None and res.pools:
                    x_, _, i_ = (example[k] for k in where)
                    out_like = torch.empty(i_.shape[0], i_.shape[1], x_.shape[1], dtype=torch.float32)
                    here = [p for p in res.pools if p.device == dev]
                    for pool in res.pools:
                        if pool.device != dev:
                            peers += [pool.device] if pool.device not in peers else []
                            # straight between the GPUs: the inputs into an inbox on the other GPU, its
                            # replies into one here, each followed by a flag in the receiver's memory
                            inbox = [torch.empty_like(example[k], device=pool.device) for k in where]
                            peer_boxes[pool.device] = (pool, inbox, torch.zeros(1, dtype=torch.int32, device=pool.device),
                                                       torch.empty_like(out_like, device=dev),
                                                       torch.zeros(1, dtype=torch.int32, device=dev),
                                                       pool.slot_cpu.to(dev))  # its slot map, here, for the combine
                    pooled[node.name] = (external.experts[layer], here, peer_boxes, where, layer)
            boxes[node.name] = ([torch.empty_like(a, device="cpu").pin_memory() if isinstance(a, torch.Tensor) else a
                                 for a in example],
                                torch.empty_like(out_like, device="cpu").pin_memory(), out_like)
        for pdev in peers:  # peer access, both ways, before any capture
            torch.empty(1, device=pdev).copy_(torch.empty(1, device=dev))
            torch.empty(1, device=dev).copy_(torch.empty(1, device=pdev))
        # per host node i: its to-CPU and from-CPU flags
        F = 2
        flags = torch.zeros(F * len(host_nodes), dtype=torch.int32, pin_memory=True)
        base = driver.device_address(flags)

        def flag(i, k):
            return base + 4 * (F * i + k)

        mailboxes = []  # per host node: (node, pinned inputs, kwargs, outbox, flag offset, pooled)
        side = torch.cuda.Stream(device=dev)  # this GPU's pools' kernels, forked from the step and joined back
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            stream = torch.cuda.current_stream()
            for node in graph.nodes:
                args_ = [resolve(r) for r in node.args]
                kwargs_ = {k: resolve(r) for k, r in node.kwargs.items()}
                if node.kind == "dense":
                    module = getattr(self.split, node.name)
                    if not isinstance(module, aoti.GraphedSegment) or module.runner is None:
                        raise RuntimeError(f"step graph: {node.name} is not a loaded AOTInductor segment")
                    produced[node.name] = module.runner(*args_)
                    continue
                i = len(mailboxes)
                pinned, outbox, like = boxes[node.name]
                moe, here, peer_boxes, where, _ = pooled.get(node.name, (None, [], {}, None, None))
                hot = []
                if here:
                    x_, w_, i_ = (args_[k] for k in where)
                    side.wait_stream(stream)
                    with torch.cuda.stream(side):
                        for pool in here:
                            slots = pool.slot_gpu[i_ - moe.experts_start_idx]
                            hot.append((slots >= 0, triton_ops.fp4_experts_decode(
                                x_, w_, slots, pool.w1, pool.s1, pool.w3, pool.s3, pool.w2, pool.s2,
                                moe.swiglu_limit, per_pair=True)))
                for p, a in zip(pinned, args_):
                    if isinstance(a, torch.Tensor):
                        p.copy_(a, non_blocking=True)
                driver.write_value(stream, flag(i, 0), 1)
                for pdev, (pool, inbox, flag_in, _, _, _) in peer_boxes.items():
                    for dst, k in zip(inbox, where):
                        driver.memcpy(stream, dst, args_[k])
                    driver.write_value(stream, flag_in.data_ptr(), 1)
                driver.wait_value(stream, flag(i, 1), 1)
                driver.write_value(stream, flag(i, 1), 0)
                reply = torch.empty_like(like, device=dev)
                reply.copy_(outbox, non_blocking=True)
                if moe is not None:  # per-pair replies: each from whoever holds the expert, then the k-order sum
                    i_ = args_[where[2]]
                    for pdev, (pool, _, _, theirs, flag_back, slot_here) in peer_boxes.items():
                        driver.wait_value(stream, flag_back.data_ptr(), 1)
                        driver.write_value(stream, flag_back.data_ptr(), 0)
                        mine = slot_here[i_ - moe.experts_start_idx] >= 0
                        reply = torch.where(mine.unsqueeze(-1), theirs, reply)
                    if hot:
                        stream.wait_stream(side)
                    for mine, replies in hot:
                        reply = torch.where(mine.unsqueeze(-1), replies, reply)
                    y = torch.zeros(reply.shape[0], reply.shape[2], dtype=torch.float32, device=dev)
                    for k in range(reply.shape[1]):
                        y = y + reply[:, k]
                    reply = y
                produced[node.name] = reply
                mailboxes.append((node, pinned, kwargs_, outbox, F * i, pooled.get(node.name)))
            output = resolve(graph.output)
        peer_graphs = [self._capture_peer(pdev, peers.index(pdev), driver, flags, F, host_nodes, boxes, pooled)
                       for pdev in peers]
        worker = _Worker(self.split, mailboxes, flags, external)  # the engine's scope, for its thread
        worker.start()
        # all at once: a capture that fails leaves the step graph as it was, and the next step tries again
        self.graph, self._output, self._peer_graphs, self._worker = g, output, peer_graphs, worker
        weakref.finalize(self, worker.stop)  # the engine dropped: the thread lets go of the model

    def _capture_peer(self, pdev, p, driver, flags, F, host_nodes, boxes, pooled):
        """Another GPU's part of the step, as its own graph: for each
        experts node with a pool there, in step order, wait for the node's
        flag in this GPU's memory, run the pool's kernels on the inbox the
        engine's GPU filled, copy the per-pair replies straight into the
        engine's GPU and raise the flag there."""
        with torch.cuda.device(pdev):
            work = []
            for node in host_nodes:
                if node.name in pooled and pdev in pooled[node.name][2]:
                    moe, _, peer_boxes, _, _ = pooled[node.name]
                    work.append((moe,) + tuple(peer_boxes[pdev][:5]))
            stream = torch.cuda.Stream(device=pdev)
            torch.cuda.synchronize(pdev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                s = torch.cuda.current_stream()
                for moe, pool, inbox, flag_in, theirs, flag_back in work:
                    driver.wait_value(s, flag_in.data_ptr(), 1)
                    driver.write_value(s, flag_in.data_ptr(), 0)
                    x_, w_, i_ = inbox
                    slots = pool.slot_gpu[i_ - moe.experts_start_idx]
                    replies = triton_ops.fp4_experts_decode(x_, w_, slots, pool.w1, pool.s1, pool.w3, pool.s3,
                                                            pool.w2, pool.s2, moe.swiglu_limit, per_pair=True)
                    driver.memcpy(s, theirs, replies)
                    driver.write_value(s, flag_back.data_ptr(), 1)
        return pdev, g


class _Worker(threading.Thread):
    """Runs the host nodes on their pinned inboxes, in step order, each
    when the GPU raises its to-CPU flag."""

    def __init__(self, split, mailboxes, flags, external):
        super().__init__(name="vxpu-step-graph-host", daemon=True)
        self.split, self.mailboxes, self.flags, self.external = split, mailboxes, flags, external
        self.view = flags.numpy()
        self.go = threading.Event()
        self.error = None
        self.stopping = False

    def stop(self):
        """Ends the thread between steps (it waits on go then, not on a flag)."""
        self.stopping = True
        self.go.set()

    def run(self):
        from . import ops
        with ops.bound(self.external):  # the host ops find the engine's banks and tables through it
            self._loop()

    def _loop(self):
        while True:
            self.go.wait()
            self.go.clear()
            if self.stopping:
                return
            try:
                for node, pinned, kwargs, outbox, f, pooled in self.mailboxes:
                    while self.view[f] != 1:
                        pass
                    self.view[f] = 0
                    if pooled is None:
                        out = getattr(self.split, node.name)(*pinned, **kwargs)
                        outbox.copy_(out[0] if isinstance(out, (tuple, list)) else out)
                    else:
                        self._answer_cold(pooled, pinned, outbox)
                    self.view[f + 1] = 1
            except BaseException as e:  # the GPU would wait forever: release it, then report
                self.error = e
                for _, _, _, _, f, _ in self.mailboxes:
                    self.view[f + 1] = 1

    def _answer_cold(self, pooled, pinned, outbox):
        """The pairs no in-graph pool holds, per pair, into the outbox
        [n, topk, dim] (zero where a pool answers, in the graph). Records
        the node as _moe_mailboxes does (ops.py), so its selections reach
        the statistics the placements come from."""
        from . import ops
        from .mailboxes import CpuOwner
        moe, here, peer_boxes, where, layer = pooled
        rec = ops._recorder()
        t0 = time.perf_counter()
        x, w, i = (pinned[k] for k in where)
        held = torch.zeros_like(i, dtype=torch.bool)
        for pool in list(here) + [box[0] for box in peer_boxes.values()]:
            held |= pool.slot_cpu[i - moe.experts_start_idx] >= 0
        cold = ~held
        outbox.zero_()
        if cold.any():
            n, topk = i.shape
            tokens = torch.arange(n).view(n, 1).expand(n, topk)[cold]
            outbox[cold] = CpuOwner().answer(moe, x, tokens, i[cold], w[cold])
        if rec is not None:
            n_cold = int(cold.sum())
            rec.host_op("moe_experts", impl_ms=(time.perf_counter() - t0) * 1e3,
                        gpu_pairs=float(i.numel() - n_cold), cpu_pairs=float(n_cold))
            rec.selected_experts(layer, i.flatten().tolist())
