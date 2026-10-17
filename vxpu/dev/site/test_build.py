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

"""Unit tests for the static results site generator."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

# Ensure the site directory is in sys.path so test_build runs standalone
sys.path.insert(0, os.path.dirname(__file__))
from build import build_site, find_runs


class TestBuildSite(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "results")

    def test_find_runs(self):
        runs = find_runs(self.fixtures_dir)
        self.assertEqual(len(runs), 3)

        labels = [r.label for r in runs]
        self.assertIn("DeepSeek-V4.1-Flash baseline", labels)
        self.assertIn("Direct host ops", labels)
        self.assertIn("Triton fp8 GEMV", labels)

        # Baseline run checks
        baseline = next(r for r in runs if r.commit == "be206b6")
        self.assertEqual(baseline.clean_ms_per_token, 51.1)
        self.assertEqual(baseline.clean_best_ms_per_token, 50.8)
        self.assertEqual(baseline.per_step_ms.get("dense"), 26.6)
        self.assertEqual(baseline.per_step_ms.get("experts"), 24.0)
        self.assertEqual(baseline.experts_routing_body_ms, 20.5)
        self.assertEqual(len(baseline.dense_segments), 4)
        self.assertEqual(len(baseline.vectors_rows), 2)
        self.assertTrue(baseline.vectors_ok)

        # Direct host ops run checks (has tie flip)
        direct = next(r for r in runs if r.commit == "c65da3c")
        self.assertEqual(direct.clean_ms_per_token, 47.3)
        self.assertTrue(direct.vectors_ok)
        sky_row = next(row for row in direct.vectors_rows if row["prompt"] == "sky")
        self.assertIsNotNone(sky_row["flip"])
        self.assertEqual(sky_row["flip"]["kind"], "tie")
        self.assertEqual(sky_row["flip"]["position"], 4)

        # Precomputed compare run checks
        gemv = next(r for r in runs if r.commit == "4a3ce9f")
        self.assertEqual(gemv.clean_ms_per_token, 45.4)
        self.assertEqual(len(gemv.vectors_rows), 2)
        self.assertTrue(gemv.vectors_ok)

    def test_build_site_from_fixture(self):
        with tempfile.TemporaryDirectory() as out_dir:
            build_site(self.fixtures_dir, out_dir)

            # Check index.html exists
            index_path = os.path.join(out_dir, "index.html")
            self.assertTrue(os.path.isfile(index_path))

            with open(index_path, "r", encoding="utf-8") as f:
                index_content = f.read()

            # Verify index elements
            self.assertIn("DeepSeek-V4.1-Flash baseline", index_content)
            self.assertIn("Direct host ops", index_content)
            self.assertIn("Triton fp8 GEMV", index_content)
            self.assertIn("be206b6", index_content)
            self.assertIn("c65da3c", index_content)
            self.assertIn("4a3ce9f", index_content)
            self.assertIn("51.1 ms", index_content)
            self.assertIn("47.3 ms", index_content)
            self.assertIn("45.4 ms", index_content)
            self.assertIn("NVIDIA RTX PRO 6000", index_content)

            # Check inline SVG chart
            self.assertIn("<svg class=\"latency-chart\"", index_content)
            self.assertIn("Median ms/token", index_content)
            self.assertIn("Best ms/token", index_content)

            # Check run pages
            for run_file in os.listdir(out_dir):
                if run_file.startswith("run-") and run_file.endswith(".html"):
                    run_path = os.path.join(out_dir, run_file)
                    with open(run_path, "r", encoding="utf-8") as f:
                        run_html = f.read()

                    # Per-step bar
                    self.assertIn("<svg class=\"step-bar\"", run_html)
                    self.assertIn("dense", run_html)
                    self.assertIn("experts", run_html)

                    # Heaviest segments
                    self.assertIn("Heaviest Segments", run_html)
                    self.assertIn("submod_0", run_html)

                    # Test vectors verification
                    self.assertIn("Test Vectors Verification", run_html)
                    self.assertIn("sky", run_html)
                    self.assertIn("kubernetes", run_html)

    def test_cli_execution(self):
        with tempfile.TemporaryDirectory() as out_dir:
            build_script = os.path.join(os.path.dirname(__file__), "build.py")
            cmd = [sys.executable, build_script, self.fixtures_dir, out_dir]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "index.html")))

    def test_single_run_without_run_json(self):
        with tempfile.TemporaryDirectory() as tmp_res_dir:
            run_dir = os.path.join(tmp_res_dir, "exp-1")
            os.makedirs(run_dir)
            attr_file = os.path.join(run_dir, "attribute_decode.json")
            with open(attr_file, "w") as f:
                json.dump({
                    "clean_ms_per_token": 42.5,
                    "clean_best_ms_per_token": 42.0,
                    "decode_steps": 16,
                    "per_step_ms": {"dense": 20.0, "experts": 22.5},
                    "dense_segments": [{"segment": "dense_0", "ms": 5.0, "nodes": 10, "ops": [["mul", 2]]}]
                }, f)

            with tempfile.TemporaryDirectory() as out_dir:
                build_site(tmp_res_dir, out_dir)
                index_path = os.path.join(out_dir, "index.html")
                self.assertTrue(os.path.isfile(index_path))
                with open(index_path, "r") as f:
                    content = f.read()
                self.assertIn("exp-1", content)
                self.assertIn("42.5 ms", content)

    def test_empty_results_directory(self):
        with tempfile.TemporaryDirectory() as tmp_res_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                build_site(tmp_res_dir, out_dir)
                index_path = os.path.join(out_dir, "index.html")
                self.assertTrue(os.path.isfile(index_path))
                with open(index_path, "r") as f:
                    content = f.read()
                self.assertIn("No latency measurements", content)


if __name__ == "__main__":
    unittest.main()
