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

"""The lowering under layer sharding: a fused body traced for a segment on
another GPU names that GPU, not the one the lowering runs on.

    python -m unittest vxpu.lower_test
"""

import unittest

import torch
import torch.fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from . import lower


def _stack(path):
    return {"model": ("model", "Model"), path: (path, "Module")}


class SegmentDevices(unittest.TestCase):
    """_segment_devices mirrors the sharding's rule for the cut: a segment
    goes to the GPU of the highest layer its nodes come from, the head
    goes where the segment before it went."""

    def test_rule(self):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        a7 = g.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
        a7.meta["nn_module_stack"] = _stack("model.layers.7.attn")
        host7 = g.call_function(torch.ops.vxpu.moe_experts.default, (a7, a7, a7, 7))  # the experts host op
        tail7 = g.call_function(torch.ops.aten.mul.Tensor, (host7, 3.0))
        tail7.meta["nn_module_stack"] = _stack("model.layers.7.ffn.shared")
        head8 = g.call_function(torch.ops.aten.mul.Tensor, (tail7, 4.0))
        head8.meta["nn_module_stack"] = _stack("model.layers.8.attn")
        host8 = g.call_function(torch.ops.vxpu.moe_experts.default, (head8, head8, head8, 8))
        head = g.call_function(torch.ops.aten.mul.Tensor, (host8, 5.0))  # the model's head: no layer
        g.output(head)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        def layer_device(layer):
            return torch.device("cuda", 0 if layer < 8 else 1)

        devices = lower._segment_devices(gm, layer_device)
        self.assertEqual(devices[a7], torch.device("cuda", 0))
        self.assertEqual(devices[tail7], torch.device("cuda", 1), "a layer's tail goes with the next layer's head")
        self.assertEqual(devices[head8], torch.device("cuda", 1))
        self.assertEqual(devices[head], torch.device("cuda", 1), "the head goes where the segment before it went")
        self.assertEqual(lower._segment_devices(gm, None), {})


class Retarget(unittest.TestCase):
    def test_literals(self):
        g = torch.fx.Graph()
        n = g.call_function(torch.ops.aten.arange.default, (4,), {"device": torch.device("cuda", 0), "dtype": torch.int64})
        m = g.call_function(torch.ops.aten._to_copy.default, (n,), {"device": "cuda:0", "dtype": torch.float32})
        k = g.call_function(torch.ops.aten.zeros.default, ([3],), {"device": torch.device("cpu")})
        g.output((m, k))
        self.assertEqual(lower._retarget(g, torch.device("cuda", 1)), 2)
        self.assertEqual(n.kwargs["device"], torch.device("cuda", 1))
        self.assertEqual(m.kwargs["device"], torch.device("cuda", 1))
        self.assertEqual(k.kwargs["device"], torch.device("cpu"), "a CPU literal is not a GPU index")


@unittest.skipUnless(torch.cuda.is_available(), "fake CUDA tensors need a CUDA build")
class InlineOnDevice(unittest.TestCase):
    """A body that derives devices from its inputs (an arange on start.device,
    type_as, a zeros on x.device), traced for a segment on cuda:1 while the
    values sit on cuda:0: every device literal in the inlined graph names
    cuda:1."""

    def test_inline(self):
        def body(s, x):
            positions = s.reshape(-1, 1) + torch.arange(4, device=s.device)
            y = (positions.float() * 2).type_as(x)
            return y, torch.zeros(3, device=x.device) + y.sum()

        with FakeTensorMode():
            s = torch.empty(2, dtype=torch.int64, device="cuda:0")
            x = torch.empty(2, 4, dtype=torch.bfloat16, device="cuda:0")
        gm = make_fx(lambda a, b: (a.reshape(-1, 1) + b.sum().to(torch.int64), b * 1), tracing_mode="fake")(s, x)
        node = next(n for n in gm.graph.nodes if n.op == "call_function" and "mul" in str(n.target))
        lower._inline(gm, node, body, (s, x), device=torch.device("cuda", 1))
        literals = [v for n in gm.graph.nodes for v in list(n.args) + list(n.kwargs.values())
                    if isinstance(v, (torch.device, str)) and str(v).startswith("cuda")]
        self.assertTrue(literals, "the body's factories and casts carry device literals")
        self.assertEqual({str(v) for v in literals}, {"cuda:1"})


if __name__ == "__main__":
    unittest.main()
