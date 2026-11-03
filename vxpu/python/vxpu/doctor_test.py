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

import json, os, sys, tempfile, unittest
from unittest.mock import MagicMock
from . import doctor


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.m = {"source": {"repo": "test/model"}, "files": {"s1": {"size": 4, "name": "f1"}, "s2": {"size": 4, "name": "f2"}}, "tensors": {"w": {"file_sha256": "s1", "offset": 0, "length": 4}}}
        self.b = {"external": {"experts": {"0": {"w1": {}}}, "engram": {"0": {"weight": "w"}}}}

    def tearDown(self): self.tmp.cleanup()

    def _write(self, p, data=b"x" * 4096):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(data)

    def test_blob_dir(self):
        self.assertEqual(doctor.check_blob_dir(self.m, {})[0], "FAIL")
        self.assertIn("f1", doctor.check_blob_dir(self.m, {})[1])
        bdir = os.path.join(self.root, "blobs")
        self._write(os.path.join(bdir, "s1"), b"1234")
        self.assertEqual(doctor.check_blob_dir(self.m, {"VXPU_BLOB_DIR": bdir})[0], "FAIL")
        self._write(os.path.join(bdir, "s2"), b"5678")
        self.assertEqual(doctor.check_blob_dir(self.m, {"VXPU_BLOB_DIR": bdir})[0], "PASS")

    def test_external_dir(self):
        self.assertEqual(doctor.check_external_dir(self.m, self.b, {})[0], "FAIL")
        edir = os.path.join(self.root, "ext")
        self._write(os.path.join(edir, "init"))
        self.assertEqual(doctor.check_external_dir(self.m, self.b, {"VXPU_EXTERNAL_DIR": edir})[0], "FAIL")
        for p in doctor._external_paths(self.m, self.b, edir): self._write(p)
        self.assertEqual(doctor.check_external_dir(self.m, self.b, {"VXPU_EXTERNAL_DIR": edir})[0], "PASS")

    def test_tokenizer(self):
        self.assertEqual(doctor.check_tokenizer(self.m, {})[0], "FAIL")
        hdir = os.path.join(self.root, "hf")
        self.assertEqual(doctor.check_tokenizer(self.m, {"HF_HOME": hdir})[0], "FAIL")
        self._write(os.path.join(hdir, "models--test--model", "snapshots", "rev", "tokenizer.json"), b"{}")
        self.assertEqual(doctor.check_tokenizer(self.m, {"HF_HOME": hdir})[0], "PASS")

    def test_environment_and_caches(self):
        env = {"HF_HUB_OFFLINE": "0", "VXPU_SCRATCH_DIR": self.root}
        self.assertEqual(doctor.check_environment(env)[0], "FAIL")
        env.update({"HF_HUB_OFFLINE": "1", **{k: "/tmp" for k in ("TMPDIR", "VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}})
        self.assertEqual(doctor.check_environment(env)[0], "FAIL")
        env.update({k: os.path.join(self.root, k) for k in ("TMPDIR", "VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")})
        self.assertEqual(doctor.check_environment(env)[0], "PASS")
        self.assertEqual(doctor.check_caches(env)[0], "FAIL")
        for k in ("VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"): os.makedirs(env[k])
        self.assertEqual(doctor.check_caches(env)[0], "PASS")

    def test_scratch_and_page_cache(self):
        sdir = os.path.join(self.root, "scratch")
        self._write(os.path.join(sdir, "aoti_stray", "f"), b"x" * 100)
        self._write(os.path.join(sdir, "experts.0.w1"), b"x" * 8192)
        self.assertEqual(doctor.check_scratch({"VXPU_SCRATCH_DIR": sdir, "TMPDIR": sdir})[0], "PASS")
        self.assertIn(doctor.check_page_cache(self.m, {"external": {"experts": {"0": {"w1": {}}}}}, {"VXPU_EXTERNAL_DIR": sdir})[0], ("PASS", "WARN"))

    def test_gpus_and_run_doctor(self):
        self._write(os.path.join(self.root, "locks", "LOCK-test"), b"1")
        self.assertEqual(doctor.check_gpus({"VXPU_LOCK_PATTERN": os.path.join(self.root, "locks", "LOCK-*")})[0], "WARN")
        fake = MagicMock()
        fake.cuda.is_available.return_value, fake.cuda.device_count.return_value, fake.cuda.mem_get_info.return_value = True, 2, (8e10, 9e10)
        sys.modules["torch"] = fake
        self.assertEqual(doctor.check_gpus({"VXPU_LOCK_PATTERN": "/nonexistent/*"})[0], "PASS")
        art = os.path.join(self.root, "art")
        self._write(os.path.join(art, "manifest.json"), json.dumps(self.m).encode())
        self.assertEqual(doctor.run_doctor(art, {}, log=lambda _: None), 1)


if __name__ == "__main__": unittest.main()
