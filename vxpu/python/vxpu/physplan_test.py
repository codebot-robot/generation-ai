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

"""Unit tests for physical plan mapping and run record schemas (#529)."""

import json
import os
import tempfile
import unittest

try:
    from . import physplan
except ImportError:
    import physplan


class PhysPlanTest(unittest.TestCase):

    def test_apply_env_on_fake_env(self):
        plan = {
            "format": physplan.PLAN_FORMAT,
            "artifact_md5": "abc123md5",
            "tree_commit": "commit_xyz",
            "sites": [
                {
                    "id": "model.layers.0.self_attn.sum",
                    "class": "sum site",
                    "shape": [128],
                    "row_counts": [1, 5, 8],
                    "strategy": "default",
                },
                {
                    "id": "layer.0.experts",
                    "class": "expert op",
                    "shape": [8, 128, 64],
                    "row_counts": [1, 5, 8],
                    "strategy": "fused",
                },
            ],
            "switches": {
                "VXPU_SPLIT_DECODE": "0",
            },
        }

        fake_env = {}
        knobs = physplan.apply_env(plan, env=fake_env)

        self.assertEqual(fake_env["VXPU_FUSED_EXPERTS"], "1")
        self.assertEqual(fake_env["VXPU_SPLIT_DECODE"], "0")
        self.assertEqual(knobs["VXPU_FUSED_EXPERTS"], "1")
        self.assertEqual(knobs["VXPU_SPLIT_DECODE"], "0")

    def test_per_expert_strategy(self):
        plan = {
            "format": physplan.PLAN_FORMAT,
            "artifact_md5": "abc",
            "tree_commit": "def",
            "sites": [
                {
                    "id": "layer.0.experts",
                    "class": "expert op",
                    "shape": [8, 128, 64],
                    "row_counts": [1, 5, 8],
                    "strategy": "per-expert",
                },
            ],
        }
        knobs = physplan.apply_env(plan)
        self.assertEqual(knobs.get("VXPU_FUSED_EXPERTS"), "0")

    def test_unknown_strategy_refused(self):
        bad_plan = {
            "format": physplan.PLAN_FORMAT,
            "artifact_md5": "abc",
            "tree_commit": "def",
            "sites": [
                {
                    "id": "model.layers.0.self_attn.sum",
                    "class": "sum site",
                    "strategy": "unsupported_magic_strategy",
                },
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(bad_plan)
        self.assertIn("unknown strategy", str(ctx.exception))

    def test_cross_class_strategy_refused(self):
        # fused is valid for expert op, but invalid for dense GEMV
        bad_plan = {
            "format": physplan.PLAN_FORMAT,
            "artifact_md5": "abc",
            "tree_commit": "def",
            "sites": [
                {
                    "id": "model.layers.0.self_attn.q_proj.weight",
                    "class": "dense GEMV",
                    "strategy": "fused",
                },
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(bad_plan)
        self.assertIn("unknown strategy", str(ctx.exception))

    def test_missing_plan_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            physplan.load_plan("/nonexistent/path/to/plan.json")

    def test_record_schema_versioned_and_round_trip(self):
        plan = {
            "format": physplan.PLAN_FORMAT,
            "artifact_md5": "md5_fixture",
            "tree_commit": "commit_fixture",
            "sites": [
                {
                    "id": "model.layers.0.self_attn.sum",
                    "class": "sum site",
                    "strategy": "default",
                },
            ],
        }
        fake_env = {}
        knobs = physplan.apply_env(plan, env=fake_env)

        record = physplan.create_run_record(
            plan=plan,
            knobs_in_effect=knobs,
            median_step_ms=18.4,
            tokens_per_step=1.0,
            per_card_peak_memory={"cuda:0": 1024 * 1024 * 500},
            build_time_s=1.23,
        )

        self.assertEqual(record["format"], physplan.RECORD_FORMAT)
        self.assertEqual(record["schema_version"], physplan.RECORD_SCHEMA_VERSION)
        self.assertEqual(record["artifact_md5"], "md5_fixture")
        self.assertEqual(record["tree_commit"], "commit_fixture")
        self.assertEqual(record["tokens_per_step"], 1.0)
        self.assertEqual(record["build_time_s"], 1.23)
        self.assertNotIn("tokens_a_step", record)
        self.assertNotIn("build_time", record)
        self.assertEqual(record["checks"], {"status": "not run"})

        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            physplan.write_run_record(record, run_path)

            parsed = physplan.parse_record(run_path)
            self.assertEqual(parsed["plan_hash"], record["plan_hash"])
            self.assertEqual(parsed["median_step_ms"], 18.4)
            self.assertEqual(parsed["checks"]["status"], "not run")

    def test_parse_record_validations(self):
        valid = {
            "format": physplan.RECORD_FORMAT,
            "schema_version": 1,
            "plan_hash": "hash_val",
            "tree_commit": "commit_val",
            "artifact_md5": "md5_val",
            "knobs_in_effect": {"VXPU_FUSED_EXPERTS": "1"},
            "median_step_ms": 20.0,
            "tokens_per_step": 1.0,
            "build_time_s": 0.5,
            "checks": {"status": "not run"},
        }
        # Valid parses cleanly
        self.assertEqual(physplan.parse_record(valid)["plan_hash"], "hash_val")

        # Wrong format
        wrong_fmt = dict(valid, format="vxpu-run-record/2")
        with self.assertRaises(ValueError):
            physplan.parse_record(wrong_fmt)

        # Wrong schema version
        wrong_ver = dict(valid, schema_version=2)
        with self.assertRaises(ValueError):
            physplan.parse_record(wrong_ver)

        # Missing plan_hash
        no_hash = dict(valid)
        del no_hash["plan_hash"]
        with self.assertRaises(ValueError):
            physplan.parse_record(no_hash)

        # Missing record file
        with self.assertRaises(FileNotFoundError):
            physplan.parse_record("/nonexistent/path/to/record.json")


if __name__ == "__main__":
    unittest.main()
