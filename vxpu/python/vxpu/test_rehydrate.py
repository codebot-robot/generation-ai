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

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Pre-mock torch before any local package import
mock_torch = MagicMock()
mock_torch.float64 = "F64"
mock_torch.float32 = "F32"
mock_torch.float16 = "F16"
mock_torch.bfloat16 = "BF16"
mock_torch.int64 = "I64"
mock_torch.int32 = "I32"
mock_torch.float8_e4m3fn = "F8_E4M3"
mock_torch.float8_e5m2 = "F8_E5M2"
sys.modules['torch'] = mock_torch
sys.modules['torch.export.passes'] = MagicMock()

# Pre-mock grpc and other server dependencies that might not be installed
sys.modules['grpc'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['transformers.models.auto.configuration_auto'] = MagicMock()

# Define a real base class for ExecutorServicer so that it can be subclassed and instantiated normally
class DummyExecutorServicer:
    pass

mock_pb2_grpc = MagicMock()
mock_pb2_grpc.ExecutorServicer = DummyExecutorServicer
sys.modules['vxpu_pb2_grpc'] = mock_pb2_grpc
sys.modules['vxpu.vxpu_pb2_grpc'] = mock_pb2_grpc

# Mock other pb imports in server.py
sys.modules['vxpu_pb2'] = MagicMock()
sys.modules['vxpu.vxpu_pb2'] = MagicMock()

# Add the package directory to PYTHONPATH
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from vxpu.rehydrate import fetch_tensor, DTYPES
from vxpu.server import ExecutorServicer


class TestRehydrateAndCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cas_dir = os.path.join(self.test_dir, "cas")

        # Create dummy tensors for testing
        self.ref = {
            "file_sha256": "dummy_file_hash",
            "offset": 10,
            "length": 16, # 8 float16 values = 16 bytes
            "dtype": "F16",
            "shape": [8],
        }
        self.files = {
            "dummy_file_hash": {
                "source": "http://example.com/dummy.bin"
            }
        }
        # 8 float16 values (each 2 bytes)
        self.dummy_data = bytes([i for i in range(16)])
        mock_torch.frombuffer.reset_mock()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("requests.get")
    def test_fetch_tensor_no_cas(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.content = self.dummy_data
        mock_get.return_value = mock_resp

        # 1. Fetch without CAS
        tensor, downloaded = fetch_tensor(self.ref, self.files, cas_dir=None)
        self.assertEqual(downloaded, 16)
        mock_get.assert_called_once_with(
            "http://example.com/dummy.bin",
            headers={"Range": "bytes=10-25"}
        )
        mock_torch.frombuffer.assert_called_once_with(
            bytearray(self.dummy_data),
            dtype="F16"
        )

    @patch("requests.get")
    def test_fetch_tensor_with_cas_caching(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.content = self.dummy_data
        mock_get.return_value = mock_resp

        # 1. Cold fetch (not in cache)
        tensor, downloaded = fetch_tensor(self.ref, self.files, cas_dir=self.cas_dir)
        self.assertEqual(downloaded, 16)
        mock_get.assert_called_once()
        mock_torch.frombuffer.assert_called_once_with(
            bytearray(self.dummy_data),
            dtype="F16"
        )

        mock_get.reset_mock()
        mock_torch.frombuffer.reset_mock()

        # 2. Warm fetch (should read from cache, downloaded = 0, no request)
        tensor2, downloaded2 = fetch_tensor(self.ref, self.files, cas_dir=self.cas_dir)
        self.assertEqual(downloaded2, 0)
        mock_get.assert_not_called()
        mock_torch.frombuffer.assert_called_once_with(
            bytearray(self.dummy_data),
            dtype="F16"
        )

    @patch("requests.get")
    def test_atomic_replace_concurrency(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.content = self.dummy_data
        mock_get.return_value = mock_resp

        # Mock os.replace to check it is actually called with a temp file under cas_dir
        original_replace = os.replace
        replace_called_with = []

        def mock_replace(src, dst):
            replace_called_with.append((src, dst))
            return original_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            tensor, downloaded = fetch_tensor(self.ref, self.files, cas_dir=self.cas_dir)
            self.assertEqual(downloaded, 16)
            self.assertEqual(len(replace_called_with), 1)
            src, dst = replace_called_with[0]
            self.assertTrue(src.startswith(self.cas_dir))
            self.assertTrue("tmp" in os.path.basename(src))
            url = self.files[self.ref["file_sha256"]]["source"]
            self.assertEqual(dst, os.path.join(self.cas_dir, hashlib.sha256(
                f"{self.ref['file_sha256']}:{self.ref['offset']}:{self.ref['length']}:{url}"
                .encode()).hexdigest()))

    def test_cache_pruning(self):
        # Create some files in cas_dir
        os.makedirs(self.cas_dir, exist_ok=True)
        files_created = []
        for i in range(5):
            path = os.path.join(self.cas_dir, f"file_{i}")
            with open(path, "wb") as f:
                f.write(b"a" * 10) # 10 bytes each
            files_created.append(path)

        # Set st_atime/st_mtime so that we can verify LRU sorting.
        # file_0 is oldest, file_4 is newest
        for i, path in enumerate(files_created):
            os.utime(path, (1000 + i, 1000 + i))

        # We have 5 files of 10 bytes = 50 bytes total.
        # Limit to 30 bytes (0.00000003 GB).
        # This should prune file_0 and file_1.
        servicer = ExecutorServicer(cas_dir=self.cas_dir, cas_max_size_gb=30 / (1024**3))
        servicer._prune_cache()

        self.assertFalse(os.path.exists(files_created[0]))
        self.assertFalse(os.path.exists(files_created[1]))
        self.assertTrue(os.path.exists(files_created[2]))
        self.assertTrue(os.path.exists(files_created[3]))
        self.assertTrue(os.path.exists(files_created[4]))


if __name__ == "__main__":
    unittest.main()
