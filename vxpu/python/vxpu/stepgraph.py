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
        torch.cuda.current_stream().synchronize()
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
        from . import aoti
        graph, runner = self.runner.graph, self.runner
        driver = _Driver()
        self._inputs = [v.clone() for v in self._input_values(args, kwargs)]
        inputs = dict(zip(graph.inputs, self._inputs))
        host_nodes = [n for n in graph.nodes if n.kind != "dense"]
        flags = torch.zeros(2 * len(host_nodes), dtype=torch.int32, pin_memory=True)
        base = driver.device_address(flags)
        mailboxes = []  # per host node: (node, pinned inputs, pinned outbox, flag index)
        produced = {}  # node -> its output tensors on the GPU, as captured
        dev = runner.device
        if dev.index is None:  # the pools name their GPU by index
            dev = torch.device(dev.type, torch.cuda.current_device())

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

        # every pinned buffer before the capture (allocating pinned memory
        # while a stream captures is not allowed), shaped like the inboxes
        # the runner's first step allocated
        from . import ops
        external = ops._active.external
        boxes, pooled = {}, {}
        for node in host_nodes:
            dev_ = runner._devices[node.name]
            example = [runner._arg(r, dev_, inputs) for r in node.args]
            like = next(t for (p, _, _), t in runner.inboxes.items() if p == node.name)
            out_like = like
            if node.kind == "experts":
                layer, where = _experts_call(getattr(self.split, node.name))
                res = getattr(external.experts.get(layer), "_vxpu_hot", None) if layer is not None else None
                pools = [p for p in (res.pools if res else []) if p.device == dev]
                if pools:  # the hot pairs on this GPU inside the graph; the CPU answers the rest per pair
                    x_, _, i_ = (example[k] for k in where)
                    out_like = torch.empty(i_.shape[0], i_.shape[1], x_.shape[1], dtype=torch.float32)
                    pooled[node.name] = (external.experts[layer], pools, where, layer)
            boxes[node.name] = ([torch.empty_like(a, device="cpu").pin_memory() if isinstance(a, torch.Tensor) else a
                                 for a in example],
                                torch.empty_like(out_like, device="cpu").pin_memory(), out_like)
        side = torch.cuda.Stream(device=dev)  # the in-graph pools' kernels, forked from the step and joined back
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
                hot = None
                if node.name in pooled:
                    moe, pools, where, _ = pooled[node.name]
                    x_, w_, i_ = (args_[k] for k in where)
                    side.wait_stream(stream)
                    with torch.cuda.stream(side):
                        hot = []
                        for pool in pools:
                            slots = pool.slot_gpu[i_ - moe.experts_start_idx]
                            hot.append((slots >= 0, triton_ops.fp4_experts_decode(
                                x_, w_, slots, pool.w1, pool.s1, pool.w3, pool.s3, pool.w2, pool.s2,
                                moe.swiglu_limit, per_pair=True)))
                for p, a in zip(pinned, args_):
                    if isinstance(a, torch.Tensor):
                        p.copy_(a, non_blocking=True)
                driver.write_value(stream, base + 4 * (2 * i), 1)
                driver.wait_value(stream, base + 4 * (2 * i + 1), 1)
                driver.write_value(stream, base + 4 * (2 * i + 1), 0)
                reply = torch.empty_like(like, device=dev)
                reply.copy_(outbox, non_blocking=True)
                if hot is not None:  # per-pair replies: the GPU's where it holds the expert, then the k-order sum
                    stream.wait_stream(side)
                    for mine, replies in hot:
                        reply = torch.where(mine.unsqueeze(-1), replies, reply)
                    y = torch.zeros(reply.shape[0], reply.shape[2], dtype=torch.float32, device=dev)
                    for k in range(reply.shape[1]):
                        y = y + reply[:, k]
                    reply = y
                produced[node.name] = reply
                mailboxes.append((node, pinned, kwargs_, outbox, i, pooled.get(node.name)))
            output = resolve(graph.output)
        self.graph, self._output = g, output
        from . import ops
        self._worker = _Worker(self.split, mailboxes, flags, ops._active.external)  # the engine's scope, for its thread
        self._worker.start()
        weakref.finalize(self, self._worker.stop)  # the engine dropped: the thread lets go of the model


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
                for node, pinned, kwargs, outbox, i, pooled in self.mailboxes:
                    while self.view[2 * i] != 1:
                        pass
                    self.view[2 * i] = 0
                    if pooled is None:
                        out = getattr(self.split, node.name)(*pinned, **kwargs)
                        outbox.copy_(out[0] if isinstance(out, (tuple, list)) else out)
                    else:
                        self._answer_cold(pooled, pinned, outbox)
                    self.view[2 * i + 1] = 1
            except BaseException as e:  # the GPU would wait forever: release it, then report
                self.error = e
                for _, _, _, _, i, _ in self.mailboxes:
                    self.view[2 * i + 1] = 1

    def _answer_cold(self, pooled, pinned, outbox):
        """The pairs no in-graph pool holds, per pair, into the outbox
        [n, topk, dim] (zero where a pool answers, in the graph). Records
        the node as _moe_mailboxes does (ops.py), so its selections reach
        the statistics the placements come from."""
        from . import ops
        from .mailboxes import CpuOwner
        moe, pools, where, layer = pooled
        rec = ops._recorder()
        t0 = time.perf_counter()
        x, w, i = (pinned[k] for k in where)
        held = torch.zeros_like(i, dtype=torch.bool)
        for pool in pools:
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
