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

"""Unit tests for memory-mapped tensor crash-safety, range coalescing, and streaming in vxpu.rehydrate."""

import collections
import hashlib
import json
import os
import tempfile
import unittest
import weakref
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

            partials = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials), 1)

            buf[0] = 42
            commit()

            self.assertTrue(os.path.exists(final_path))
            self.assertEqual(os.path.getsize(final_path), length)
            partials_after = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials_after), 0)

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

            self.assertGreaterEqual(len(fsynced_fds), 2)
            self.assertTrue(os.path.exists(final_path))

    def test_file_backed_concurrent_loaders_exclusive_temps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 256

            buf1, fresh1, commit1 = rehydrate.file_backed(final_path, length)
            buf2, fresh2, commit2 = rehydrate.file_backed(final_path, length)

            self.assertTrue(fresh1)
            self.assertTrue(fresh2)

            partials = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials), 2)

            buf1[:] = 1
            buf2[:] = 2

            commit1()
            self.assertTrue(os.path.exists(final_path))
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(), b"\x01" * length)

            commit2()
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(), b"\x02" * length)

    def test_file_backed_interrupted_leaves_no_final_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 256

            # Simulate load interrupted before commit
            buf, fresh, _ = rehydrate.file_backed(final_path, length)
            self.assertTrue(fresh)
            buf[0] = 123

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

            with open(final_path, "wb") as f:
                f.write(b"\x07" * length)

            stray_path = os.path.join(tmpdir, "tensor.bin.partial.old123")
            with open(stray_path, "wb") as f:
                f.write(b"garbage")

            self.assertTrue(os.path.exists(stray_path))

            buf, fresh, _ = rehydrate.file_backed(final_path, length)
            self.assertFalse(fresh)
            self.assertEqual(buf[0].item(), 7)
            self.assertFalse(os.path.exists(stray_path))

    def test_file_backed_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "tensor.bin")
            length = 64

            with rehydrate.file_backed(final_path, length) as fb:
                self.assertTrue(fb.fresh)
                fb.tensor[0] = 55

            self.assertTrue(os.path.exists(final_path))
            partials = [f for f in os.listdir(tmpdir) if f.startswith("tensor.bin.partial.")]
            self.assertEqual(len(partials), 0)

            fail_path = os.path.join(tmpdir, "fail.bin")
            with self.assertRaises(RuntimeError):
                with rehydrate.file_backed(fail_path, length) as fb:
                    fb.tensor[0] = 77
                    raise RuntimeError("simulated crash during fill")

            self.assertFalse(os.path.exists(fail_path))
            partials_fail = [f for f in os.listdir(tmpdir) if f.startswith("fail.bin.partial.")]
            self.assertEqual(len(partials_fail), 1)

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

            def failing_read_range(*args, **kwargs):
                raise RuntimeError("simulated network failure mid-fill")

            with mock.patch.object(rehydrate, "_read_range", side_effect=failing_read_range):
                with self.assertRaises(RuntimeError):
                    rehydrate.fetch_tensor(ref, files, mmap_dir=mmap_dir)

            self.assertFalse(os.path.exists(final_path))
            partials = [f for f in os.listdir(mmap_dir) if f.startswith(f"{key}.partial")]
            self.assertGreaterEqual(len(partials), 1)

            t, dl = rehydrate.fetch_tensor(ref, files, mmap_dir=mmap_dir)
            self.assertEqual(dl, len(data))
            self.assertTrue(os.path.exists(final_path))
            self.assertEqual(bytes(t.numpy()), data)

            t2, dl2 = rehydrate.fetch_tensor(ref, files, mmap_dir=mmap_dir)
            self.assertEqual(dl2, 0)
            self.assertEqual(bytes(t2.numpy()), data)

    def test_fetch_stacked_crash_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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

            with mock.patch.object(rehydrate, "fetch_tensor", side_effect=RuntimeError("part fetch failed")):
                with self.assertRaises(RuntimeError):
                    rehydrate.fetch_stacked(spec, manifest_tensors, files, mmap_path=mmap_path)

            self.assertFalse(os.path.exists(mmap_path))
            bank_dir = os.path.dirname(mmap_path)
            partials = [f for f in os.listdir(bank_dir) if f.startswith("expert.bin.partial")]
            self.assertGreaterEqual(len(partials), 1)

            out, dl = rehydrate.fetch_stacked(spec, manifest_tensors, files, mmap_path=mmap_path)
            self.assertEqual(dl, 128)
            self.assertTrue(os.path.exists(mmap_path))
            self.assertEqual(bytes(out[0].numpy()), part0)
            self.assertEqual(bytes(out[1].numpy()), part1)

            out2, dl2 = rehydrate.fetch_stacked(spec, manifest_tensors, files, mmap_path=mmap_path)
            self.assertEqual(dl2, 0)
            self.assertEqual(bytes(out2[0].numpy()), part0)
            self.assertEqual(bytes(out2[1].numpy()), part1)


class TestCoalesceAndStreaming(unittest.TestCase):
    def test_build_part_spec(self):
        class DummyRun:
            def __init__(self, offset, length, items):
                self.offset = offset
                self.length = length
                self.items = items

        t1 = torch.empty(10, dtype=torch.uint8)
        t2 = torch.empty(10, dtype=torch.uint8)
        i1 = rehydrate.ReadItem("sha", 100, 10, t1)
        i2 = rehydrate.ReadItem("sha", 120, 10, t2)
        run = DummyRun(100, 30, [i1, i2])

        spec, touched = rehydrate.build_part_spec(run, 100, 30)
        self.assertEqual(len(spec), 3)
        self.assertEqual(spec[0], (i1, 0, 10, False))
        self.assertEqual(spec[1], (None, 0, 10, True))  # gap
        self.assertEqual(spec[2], (i2, 0, 10, False))
        self.assertEqual(touched, [i1, i2])

        spec_sub, touched_sub = rehydrate.build_part_spec(run, 105, 10)
        self.assertEqual(len(spec_sub), 2)
        self.assertEqual(spec_sub[0], (i1, 5, 5, False))
        self.assertEqual(spec_sub[1], (None, 0, 5, True))  # gap
        self.assertEqual(touched_sub, [i1])

    def test_plan_and_fetch_coalesce_and_streaming_demux(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = bytes(i % 256 for i in range(2 * 1024 * 1024))
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            dst1 = torch.empty(250, dtype=torch.int32)
            dst2 = torch.empty((2, 500), dtype=torch.uint8)
            dst3 = torch.empty(1000, dtype=torch.uint8)

            item1 = rehydrate.ReadItem(file_sha, 1000, 1000, dst1)
            item2_row0 = rehydrate.ReadItem(file_sha, 2500, 500, dst2[0], cat_dim=0, slot_offset=0, shape=[500], dtype=torch.uint8)
            item2_row1 = rehydrate.ReadItem(file_sha, 3000, 500, dst2[1], cat_dim=0, slot_offset=0, shape=[500], dtype=torch.uint8)
            item3 = rehydrate.ReadItem(file_sha, 1500000, 1000, dst3)

            downloaded, requests = rehydrate.plan_and_fetch_ranges(
                [item1, item2_row0, item2_row1, item3], files, gap_bytes=rehydrate.GAP_BYTES
            )

            self.assertEqual(requests, 2)
            self.assertEqual(downloaded, 2500 + 1000)
            self.assertEqual(bytes(dst1.view(torch.uint8).numpy()), content[1000:2000])
            self.assertEqual(bytes(dst2[0].numpy()), content[2500:3000])
            self.assertEqual(bytes(dst2[1].numpy()), content[3000:3500])
            self.assertEqual(bytes(dst3.numpy()), content[1500000:1501000])

    def test_cpu_tensor_released_on_completion(self):
        """Verify that CPU tensors are freed as items complete (weakref becomes None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = b"A" * 1024
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            cpu_tensor = torch.empty(1024, dtype=torch.uint8)
            shared = {"t": cpu_tensor}
            wref = weakref.ref(cpu_tensor)
            del cpu_tensor  # shared["t"] is now the only reference

            def on_complete():
                # Replaces shared entry with device copy, dropping the reference
                shared["t"] = shared["t"].clone()

            item = rehydrate.ReadItem(file_sha, 0, 1024, shared["t"], on_complete=on_complete)

            self.assertIsNotNone(wref())
            rehydrate.plan_and_fetch_ranges([item], files)

            # After completion, item drops target reference, so wref() should be None!
            self.assertIsNone(wref())
            self.assertEqual(bytes(shared["t"].numpy()), content)

    def test_cat_dim_staging_buffer_freed_after_copy(self):
        """Verify that non-contiguous cat_dim slices allocate staging buffers only at fetch time and free them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = bytes(range(16))
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            target = torch.zeros((2, 4), dtype=torch.uint8)
            item0 = rehydrate.ReadItem(file_sha, 0, 4, target, cat_dim=1, slot_offset=0, shape=[2, 2], dtype=torch.uint8)
            item1 = rehydrate.ReadItem(file_sha, 4, 4, target, cat_dim=1, slot_offset=2, shape=[2, 2], dtype=torch.uint8)

            self.assertFalse(item0.is_contiguous_target())
            self.assertFalse(item1.is_contiguous_target())

            rehydrate.plan_and_fetch_ranges([item0, item1], files)

            self.assertEqual(bytes(target[:, 0:2].contiguous().numpy()), content[0:4])
            self.assertEqual(bytes(target[:, 2:4].contiguous().numpy()), content[4:8])
            self.assertIsNone(item0.target)
            self.assertIsNone(item1.target)

    def test_zero_length_items_complete_at_planning_time(self):
        file_sha = "0000000000000000000000000000000000000000000000000000000000000000"
        files = {file_sha: {"source": "file:///nonexistent", "size": 0}}

        completed = []
        dst_empty = torch.empty(0, dtype=torch.float32)
        item_empty = rehydrate.ReadItem(file_sha, 0, 0, dst_empty, shape=[0], dtype=torch.float32,
                                        on_complete=lambda: completed.append(True))

        dl, req = rehydrate.plan_and_fetch_ranges([item_empty], files)
        self.assertEqual(dl, 0)
        self.assertEqual(req, 0)
        self.assertEqual(completed, [True])

    def test_overlapping_items_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = b"0123456789"
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            # 1. Exact duplicates: copy from primary
            dstA = torch.zeros(10, dtype=torch.uint8)
            dstB = torch.zeros(10, dtype=torch.uint8)
            itemA = rehydrate.ReadItem(file_sha, 0, 10, dstA)
            itemB = rehydrate.ReadItem(file_sha, 0, 10, dstB)
            dl, req = rehydrate.plan_and_fetch_ranges([itemA, itemB], files)
            self.assertEqual(req, 1)
            self.assertEqual(dl, 10)
            self.assertEqual(bytes(dstA.numpy()), content)
            self.assertEqual(bytes(dstB.numpy()), content)

            # 2. Partial overlap: should raise ValueError
            dstC = torch.zeros(8, dtype=torch.uint8)
            dstD = torch.zeros(8, dtype=torch.uint8)
            itemC = rehydrate.ReadItem(file_sha, 0, 8, dstC)
            itemD = rehydrate.ReadItem(file_sha, 4, 8, dstD)
            with self.assertRaises(ValueError) as ctx:
                rehydrate.plan_and_fetch_ranges([itemC, itemD], files)
            self.assertIn("overlapping items", str(ctx.exception))

    def test_cas_hits_do_not_delay_network_misses(self):
        """Verify that network tasks are fetched first before CAS hits are read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            cas_dir = os.path.join(tmpdir, "cas")
            os.makedirs(cas_dir, exist_ok=True)
            content = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            key_hit = hashlib.sha256(f"{file_sha}:0:10".encode()).hexdigest()
            with open(os.path.join(cas_dir, key_hit), "wb") as f:
                f.write(content[:10])

            dst_hit = torch.zeros(10, dtype=torch.uint8)
            dst_miss = torch.zeros(10, dtype=torch.uint8)
            item_hit = rehydrate.ReadItem(file_sha, 0, 10, dst_hit)
            item_miss = rehydrate.ReadItem(file_sha, 10, 10, dst_miss)

            order = []
            orig_read_range = rehydrate._read_range
            orig_read_cas = rehydrate._read_cas_item

            def tracked_read_range(*args, **kwargs):
                order.append("net")
                return orig_read_range(*args, **kwargs)

            def tracked_read_cas(*args, **kwargs):
                order.append("cas")
                return orig_read_cas(*args, **kwargs)

            with mock.patch.object(rehydrate, "_read_range", side_effect=tracked_read_range), \
                 mock.patch.object(rehydrate, "_read_cas_item", side_effect=tracked_read_cas):
                rehydrate.plan_and_fetch_ranges([item_hit, item_miss], files, cas_dir=cas_dir, workers=1)

            self.assertEqual(order[0], "net")
            self.assertEqual(bytes(dst_hit.numpy()), content[:10])
            self.assertEqual(bytes(dst_miss.numpy()), content[10:20])

    def test_unaligned_storage_offsets_and_wider_dtypes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = bytes(i % 256 for i in range(1024))
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            dst1 = torch.empty(3, dtype=torch.uint8)
            dst2 = torch.empty(1, dtype=torch.int32)
            item1 = rehydrate.ReadItem(file_sha, 0, 3, dst1)
            item2 = rehydrate.ReadItem(file_sha, 3, 4, dst2)

            dl, req = rehydrate.plan_and_fetch_ranges([item1, item2], files)
            self.assertEqual(req, 1)
            self.assertEqual(dl, 7)
            self.assertEqual(bytes(dst1.numpy()), content[0:3])
            self.assertEqual(bytes(dst2.view(torch.uint8).numpy()), content[3:7])

    def test_run_part_splitting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = b"X" * 1000
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            dst = torch.empty(1000, dtype=torch.uint8)
            item = rehydrate.ReadItem(file_sha, 0, 1000, dst)

            dl, req = rehydrate.plan_and_fetch_ranges([item], files, part_bytes=200)
            self.assertEqual(req, 5)
            self.assertEqual(dl, 1000)
            self.assertEqual(bytes(dst.numpy()), content)

    def test_engine_host_memory_bounded_during_load(self):
        """Test at Engine level that peak host memory residency is bounded and CPU tensors are released."""
        from .engine import Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            art_dir = os.path.join(tmpdir, "artifact")
            os.makedirs(art_dir, exist_ok=True)

            # Create distinct files of 1024 bytes
            files = {}
            for i in range(32):
                fpath = os.path.join(tmpdir, f"file_{i}.bin")
                data = bytes([i % 256] * 1024)
                with open(fpath, "wb") as f:
                    f.write(data)
                fsha = hashlib.sha256(data).hexdigest()
                files[fsha] = {"source": f"file://{fpath}", "size": 1024}

            file_keys = list(files.keys())

            # 16 bound tensors across the first 16 files
            bound = {}
            tensors_meta = {}
            for i in range(16):
                fsha = file_keys[i]
                bound[f"model.layer.{i}.weight"] = {
                    "file_sha256": fsha,
                    "offset": 0,
                    "length": 512,
                    "dtype": "U8",
                    "shape": [512],
                }

            # 2 stacked expert banks across the next 16 files (each bank has 4 rows of 2 parts = 8 parts)
            stacked = {}
            for b in range(2):
                parts = []
                for r in range(4):
                    row_parts = []
                    for p in range(2):
                        tname = f"expert_b{b}_r{r}_p{p}"
                        fsha = file_keys[16 + (b * 8 + r * 2 + p)]
                        tensors_meta[tname] = {
                            "file_sha256": fsha,
                            "offset": 0,
                            "length": 256,
                            "dtype": "U8",
                            "shape": [256],
                        }
                        row_parts.append(tname)
                    parts.append(row_parts)
                stacked[f"model.bank.{b}"] = {
                    "parts": parts,
                    "cat_dim": 0,
                    "stack": True,
                }

            local_repo = os.path.join(tmpdir, "local_repo")
            os.makedirs(local_repo, exist_ok=True)
            manifest = {
                "config": {"model_type": "llama", "eos_token_id": 2},
                "source": {"repo": local_repo},
                "files": files,
                "tensors": tensors_meta,
            }
            binding = {
                "bound": bound,
                "stacked": stacked,
                "external": {},
                "derived": [],
                "unused": [],
                "state": [],
            }

            with open(os.path.join(art_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            with open(os.path.join(art_dir, "binding.json"), "w") as f:
                json.dump(binding, f)

            # Track all CPU tensors created during the load
            cpu_tensor_refs = []
            orig_torch_empty = torch.empty

            def tracking_empty(*args, **kwargs):
                t = orig_torch_empty(*args, **kwargs)
                if t.device.type == "cpu" and t.numel() > 0:
                    cpu_tensor_refs.append(weakref.ref(t))
                return t

            # Mock export load (dummy decode.pt2)
            class DummyProbe:
                state_dict = {}
                constants = {}

            def fake_move(self, fqn, dev):
                self.shared[fqn] = torch.empty(0)

            with mock.patch("torch.export.load", return_value=DummyProbe()), \
                 mock.patch.object(Engine, "_build", return_value=None), \
                 mock.patch.object(Engine, "_move_to_device", fake_move), \
                 mock.patch("torch.empty", side_effect=tracking_empty):
                # Run Engine load for simulated device "cuda:0"
                eng = Engine(art_dir, device="cuda:0")

            # Verify that requests were issued and coalesced
            self.assertGreater(eng.requests_issued, 0)
            self.assertGreater(eng.bytes_fetched, 0)

            # ALL CPU tensors for bound and stacked weights must have been freed!
            alive_cpu = [w for w in cpu_tensor_refs if w() is not None]
            self.assertEqual(len(alive_cpu), 0, f"{len(alive_cpu)} CPU tensors are still alive after Engine load!")


if __name__ == "__main__":
    unittest.main()
