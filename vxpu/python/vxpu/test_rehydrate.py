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

"""Unit tests for memory-mapped tensor crash-safety in vxpu.rehydrate."""

import hashlib
import os
import tempfile
import unittest
from unittest import mock

import torch

from . import rehydrate


class TestCrashSafeRehydrate(unittest.TestCase):
    def test_file_backed_normal_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 512

            buf, fresh, commit = rehydrate.file_backed(final_path, length)
            self.assertTrue(fresh)
            self.assertFalse(os.path.exists(final_path))

            # A temporary file starting with tensor.bin.partial. was created
            partials = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials), 1)

            # Fill buffer
            buf[0] = 42
            commit()

            # Temp file was renamed to final_path
            self.assertTrue(os.path.exists(final_path))
            self.assertEqual(os.path.getsize(final_path), length)
            partials_after = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials_after), 0)

            # Second call sees existing complete file
            buf2, fresh2, commit2 = rehydrate.file_backed(final_path, length)
            self.assertFalse(fresh2)
            self.assertEqual(buf2[0].item(), 42)
            commit2()  # no-op

    def test_file_backed_fsync_on_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 256

            buf, fresh, commit = rehydrate.file_backed(final_path, length)
            self.assertTrue(fresh)

            fsynced_fds = []
            orig_fsync = os.fsync

            def tracking_fsync(fd):
                fsynced_fds.append(fd)
                return orig_fsync(fd)

            with mock.patch("os.fsync", side_effect=tracking_fsync):
                buf[0] = 17
                commit()

            # Verify fsync was called at least twice (once for temp file, once for directory)
            self.assertGreaterEqual(len(fsynced_fds), 2)
            self.assertTrue(os.path.exists(final_path))

    def test_file_backed_concurrent_loaders_exclusive_temps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 256

            # Two loaders start concurrently for the same missing path
            buf1, fresh1, commit1 = rehydrate.file_backed(final_path, length)
            buf2, fresh2, commit2 = rehydrate.file_backed(final_path, length)

            self.assertTrue(fresh1)
            self.assertTrue(fresh2)

            # Each loader must have its own distinct temporary file
            partials = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials), 2)

            # Loader 1 fills with 1s, Loader 2 fills with 2s
            buf1[:] = 1
            buf2[:] = 2

            # Neither clobbered the other
            self.assertEqual(buf1[0].item(), 1)
            self.assertEqual(buf2[0].item(), 2)

            # Loader 1 commits
            commit1()
            self.assertTrue(os.path.exists(final_path))
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(1), b"\x01")

            # Loader 2 commits (last rename wins)
            commit2()
            self.assertTrue(os.path.exists(final_path))
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(1), b"\x02")

    def test_file_backed_interrupted_leaves_no_final_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 256

            # Simulate load interrupted before commit
            buf, fresh, _ = rehydrate.file_backed(final_path, length)
            self.assertTrue(fresh)
            buf[0] = 123
            # Interrupted: commit() is never called

            # Temp file exists, but final_path does not
            self.assertFalse(os.path.exists(final_path))
            partials = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials), 1)

            # Next load sees it as fresh
            buf2, fresh2, commit2 = rehydrate.file_backed(final_path, length)
            self.assertTrue(fresh2)
            buf2[0] = 99
            commit2()

            self.assertTrue(os.path.exists(final_path))
            self.assertEqual(buf2[0].item(), 99)

    def test_file_backed_stray_partial_removed_when_complete_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 128

            # Create complete file at final_path
            with open(final_path, "wb") as f:
                f.write(b"\x07" * length)

            # Stray .partial files left from previous crashed retries
            legacy_partial = final_path + ".partial"
            stray_partial = final_path + ".partial.stray123"
            with open(legacy_partial, "wb") as f:
                f.write(b"garbage1")
            with open(stray_partial, "wb") as f:
                f.write(b"garbage2")

            self.assertTrue(os.path.exists(legacy_partial))
            self.assertTrue(os.path.exists(stray_partial))

            buf, fresh, _ = rehydrate.file_backed(final_path, length)
            self.assertFalse(fresh)
            self.assertEqual(buf[0].item(), 7)

            # Stray .partial files should have been removed
            self.assertFalse(os.path.exists(legacy_partial))
            self.assertFalse(os.path.exists(stray_partial))

    def test_fetch_tensor_crash_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = b"0123456789abcdef" * 16  # 256 bytes
            data_file = os.path.join(tmpdir, "source.bin")
            with open(data_file, "wb") as f:
                f.write(data)

            file_sha = hashlib.sha256(data).hexdigest()
            files = {file_sha: {"source": f"file://{data_file}", "size": len(data)}}
            ref = {
                "file_sha256": file_sha,
                "offset": 0,
                "length": len(data),
                "dtype": "U8",
                "shape": [len(data)],
            }
            mmap_dir = os.path.join(tmpdir, "mmap")
            key = hashlib.sha256(f"{file_sha}:0:{len(data)}".encode()).hexdigest()
            final_path = os.path.join(mmap_dir, key)

            # 1. Simulate failure during fill
            def failing_read_range(*args, **kwargs):
                raise RuntimeError("simulated network failure mid-fill")

            with mock.patch.object(rehydrate, "_read_range", side_effect=failing_read_range):
                with self.assertRaises(RuntimeError):
                    rehydrate.fetch_tensor(ref, files, mmap_dir=mmap_dir)

            # Leaves NO file at path; leaves stray .partial file
            self.assertFalse(os.path.exists(final_path))
            partials = [f for f in os.listdir(mmap_dir) if f.startswith(f"{key}.partial")]
            self.assertGreaterEqual(len(partials), 1)

            # 2. Next load refetches and completes
            t, dl = rehydrate.fetch_tensor(ref, files, mmap_dir=mmap_dir)
            self.assertEqual(dl, len(data))
            self.assertTrue(os.path.exists(final_path))
            self.assertEqual(bytes(t.numpy()), data)

            # 3. Completed file is reused without refetching
            t2, dl2 = rehydrate.fetch_tensor(ref, files, mmap_dir=mmap_dir)
            self.assertEqual(dl2, 0)
            self.assertEqual(bytes(t2.numpy()), data)

    def test_fetch_stacked_crash_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two parts of 64 bytes each
            part0 = b"A" * 64
            part1 = b"B" * 64
            src_file = os.path.join(tmpdir, "parts.bin")
            with open(src_file, "wb") as f:
                f.write(part0 + part1)

            file_sha = hashlib.sha256(part0 + part1).hexdigest()
            files = {file_sha: {"source": f"file://{src_file}", "size": 128}}
            manifest_tensors = {
                "t0": {"file_sha256": file_sha, "offset": 0, "length": 64, "dtype": "U8", "shape": [64]},
                "t1": {"file_sha256": file_sha, "offset": 64, "length": 64, "dtype": "U8", "shape": [64]},
            }
            spec = {
                "parts": [["t0"], ["t1"]],
                "cat_dim": 0,
                "stack": True,
            }
            mmap_path = os.path.join(tmpdir, "mmap_bank", "expert.bin")

            # 1. Simulate failure during part fetch
            with mock.patch.object(rehydrate, "fetch_tensor", side_effect=RuntimeError("part fetch failed")):
                with self.assertRaises(RuntimeError):
                    rehydrate.fetch_stacked(spec, manifest_tensors, files, mmap_path=mmap_path)

            self.assertFalse(os.path.exists(mmap_path))
            bank_dir = os.path.dirname(mmap_path)
            partials = [f for f in os.listdir(bank_dir) if f.startswith("expert.bin.partial")]
            self.assertGreaterEqual(len(partials), 1)

            # 2. Next load refetches and completes
            out, dl = rehydrate.fetch_stacked(spec, manifest_tensors, files, mmap_path=mmap_path)
            self.assertEqual(dl, 128)
            self.assertTrue(os.path.exists(mmap_path))
            self.assertEqual(bytes(out[0].numpy()), part0)
            self.assertEqual(bytes(out[1].numpy()), part1)

            # 3. Completed file is reused without refetching
            out2, dl2 = rehydrate.fetch_stacked(spec, manifest_tensors, files, mmap_path=mmap_path)
            self.assertEqual(dl2, 0)
            self.assertEqual(bytes(out2[0].numpy()), part0)
            self.assertEqual(bytes(out2[1].numpy()), part1)


if __name__ == "__main__":
    unittest.main()
