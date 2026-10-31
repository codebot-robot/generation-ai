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

"""Unit tests for the kernel plan, table validation, and record checks.

    python -m unittest vxpu.plan_test
"""

import io
import json
import unittest
import warnings
from contextlib import redirect_stdout

from . import plan
from .plan import ConfigError, check_record, describe_mismatches, load_table, table_entry

# Register mock implementations for test ops so load_table validates them
# without needing torch/ops.py to be loaded.
if "vxpu::fp8_linear" not in plan.REGISTRY:
    @plan.implementation("vxpu::fp8_linear", "reference", ("cpu",), "bit-identical", "reference")
    def _dummy_fp8_ref(*args): pass

    @plan.implementation("vxpu::fp8_linear", "cpu-kernel", ("cpu",), "rounding", "cpu kernel")
    def _dummy_fp8_cpu(*args): pass

if "vxpu::rotary" not in plan.REGISTRY:
    @plan.implementation("vxpu::rotary", "reference", ("cpu",), "bit-identical", "reference")
    def _dummy_rotary_ref(*args): pass


class PlanRecordCheck(unittest.TestCase):
    """The engine checks what each site launched against the table at load,
    and refuses a mismatch (issue #495)."""

    def setUp(self):
        self.table_data = {
            "format": "vxpu-kernel-table/1",
            "sites": [
                {"op": "vxpu::fp8_linear", "site": "model.layers.1.engram.wkv", "impl": "reference"},
                {"op": "vxpu::fp8_linear", "shapes": "1x8->16", "impl": "reference"},
                {"op": "vxpu::rotary", "impl": "reference"},
            ],
        }
        self.table = load_table(json.dumps(self.table_data))
        self.matching_record = {
            "format": "vxpu-kernel-record/1",
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "layer": 1,
                    "chosen": "reference",
                    "ran": "reference",
                    "source": "table",
                },
                {
                    "op": "vxpu::rotary",
                    "site": "model.layers.0.attn.rotary",
                    "shapes": "1x8->8",
                    "layer": 0,
                    "chosen": "reference",
                    "ran": "reference",
                    "source": "table",
                },
            ],
        }

    def test_matching_record_passes_and_summarizes(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bad = check_record(self.table, self.matching_record)
        self.assertEqual(bad, [])
        output = buf.getvalue()
        self.assertIn("[vxpu] kernel table: all 2 sites matched the table", output)

    def test_mismatched_record_raises_config_error_naming_site(self):
        mismatched = {
            "format": "vxpu-kernel-record/1",
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "layer": 1,
                    "chosen": "cpu-kernel",
                    "ran": "cpu-kernel",
                    "source": "table",
                }
            ],
        }
        with self.assertRaises(ConfigError) as ctx:
            check_record(self.table, mismatched)
        msg = str(ctx.exception)
        self.assertIn("model.layers.1.engram.wkv", msg)
        self.assertIn("ran cpu-kernel", msg)
        self.assertIn("the table names reference", msg)

    def test_mismatched_record_with_raise_false_returns_bad(self):
        mismatched = {
            "format": "vxpu-kernel-record/1",
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "chosen": "cpu-kernel",
                }
            ],
        }
        bad = check_record(self.table, mismatched, raise_on_mismatch=False)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0][1], "reference")
        described = describe_mismatches(bad, self.table)
        self.assertIn("model.layers.1.engram.wkv", described)
        self.assertIn("ran cpu-kernel", described)

    def test_no_table_warns_on_mismatch_against_chosen(self):
        no_table_record = {
            "format": "vxpu-kernel-record/1",
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "chosen": "reference",
                    "ran": "cpu-kernel",
                }
            ],
        }
        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as ws, redirect_stdout(buf):
            warnings.simplefilter("always")
            bad = check_record(None, no_table_record)
            self.assertEqual(len(bad), 1)
            self.assertEqual(len(ws), 1)
            self.assertIn("model.layers.1.engram.wkv", str(ws[0].message))
            self.assertIn("ran cpu-kernel", str(ws[0].message))
            self.assertIn("the lowering chose reference", str(ws[0].message))
        self.assertIn("warning:", buf.getvalue())

    def test_no_table_matching_emits_no_warnings(self):
        no_table_record = {
            "format": "vxpu-kernel-record/1",
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "chosen": "reference",
                    "ran": "reference",
                }
            ],
        }
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            bad = check_record(None, no_table_record)
            self.assertEqual(bad, [])
            self.assertEqual(len(ws), 0)


if __name__ == "__main__":
    unittest.main()
