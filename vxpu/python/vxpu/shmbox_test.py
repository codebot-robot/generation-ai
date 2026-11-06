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

"""GPU-free tests and round-trip benchmarks for the shared-memory control plane (#539)."""

import multiprocessing as mp
import os
import struct
import time
import unittest

from .shmbox import (
    BROADCAST,
    COORDINATOR,
    DEFAULT_SLOT_SIZE,
    FLAG_JSON,
    FLAG_RAW,
    MAGIC,
    VERSION,
    AcceptanceControl,
    ControlPlane,
    RankStatus,
    StepControl,
)


def _unique_shm_name() -> str:
    """Generates a short POSIX shared-memory name that fits within macOS 31-character limits."""
    return f"/vx_{os.getpid()}_{int(time.monotonic() * 1e6) % 1000000}"


def _worker_rank_barrier(rank: int, num_ranks: int, rounds: int, shm_name: str) -> None:
    control_plane = ControlPlane.attach(name=shm_name, rank=rank)
    try:
        for _ in range(rounds):
            control_plane.rank_barrier(timeout=5.0)
    finally:
        control_plane.close()


def _worker_all_barrier(participant: int, num_ranks: int, rounds: int, shm_name: str) -> None:
    rank = COORDINATOR if participant == num_ranks else participant
    control_plane = ControlPlane.attach(name=shm_name, rank=rank)
    try:
        for _ in range(rounds):
            control_plane.barrier(timeout=5.0)
    finally:
        control_plane.close()


def _fake_rank_step_loop(rank: int, num_ranks: int, step_count: int, shm_name: str) -> None:
    control_plane = ControlPlane.attach(name=shm_name, rank=rank)
    try:
        for step in range(1, step_count + 1):
            # 1. Receive coordinator step broadcast
            step_control = control_plane.read_step(expected_step_id=step, timeout=5.0)
            if step_control.step_id != step:
                raise ValueError(f"Rank {rank}: expected step {step}, got {step_control.step_id}")

            # 2. Rank 0 publishes acceptance choices, peers consume
            if rank == 0:
                control_plane.publish_acceptance(step_id=step, accepted_len=4, accepted_tokens=(100, 101, 102, 103))
            else:
                acceptance = control_plane.read_acceptance(expected_step_id=step, timeout=5.0)
                if acceptance.step_id != step:
                    raise ValueError(f"Rank {rank}: expected acceptance step {step}, got {acceptance.step_id}")

            # 3. Step barrier across the card ranks
            control_plane.rank_barrier(timeout=5.0)

            # 4. Report status to coordinator
            control_plane.post_status(step_id=step, status_code=0)
    finally:
        control_plane.close()


def _attaching_worker(shm_name: str) -> None:
    attached = ControlPlane.attach(name=shm_name, rank=0)
    attached.close()


class ControlPlaneTest(unittest.TestCase):
    """Tests for shared-memory mailboxes, sequence numbers, and barriers."""

    def test_total_bytes_and_layout(self) -> None:
        total = ControlPlane.total_bytes(num_ranks=4, slot_size=DEFAULT_SLOT_SIZE)
        self.assertGreater(total, 0)
        self.assertEqual(total % 4096, 0)

    def test_create_and_header(self) -> None:
        control_plane = ControlPlane.create(num_ranks=4, slot_size=1024)
        try:
            magic = bytes(control_plane.buf[0:4])
            self.assertEqual(magic, MAGIC)
            version, n_ranks, slot_sz, _ = struct.unpack_from("<IIIQ", control_plane.buf, 4)
            self.assertEqual(version, VERSION)
            self.assertEqual(n_ranks, 4)
            self.assertEqual(slot_sz, 1024)
        finally:
            control_plane.close()

    def test_invalid_participant_ids(self) -> None:
        """Verifies that out-of-range participant IDs and destinations are rejected."""
        control_plane = ControlPlane.create(num_ranks=4, slot_size=512, rank=COORDINATOR)
        try:
            # Participant rank validation in constructor
            with self.assertRaises(ValueError):
                ControlPlane(buf=control_plane.buf, rank=4, num_ranks=4)  # rank == num_ranks
            with self.assertRaises(ValueError):
                ControlPlane(buf=control_plane.buf, rank=5, num_ranks=4)  # rank > num_ranks
            with self.assertRaises(ValueError):
                ControlPlane(buf=control_plane.buf, rank=-5, num_ranks=4)  # negative non-coordinator

            # Destination and sender validation in slot_offset
            with self.assertRaises(ValueError):
                control_plane.slot_offset(src=COORDINATOR, dst=4)  # dst == num_ranks
            with self.assertRaises(ValueError):
                control_plane.slot_offset(src=COORDINATOR, dst=-5)  # invalid negative dst
            with self.assertRaises(ValueError):
                control_plane.slot_offset(src=4, dst=0)  # src == num_ranks
        finally:
            control_plane.close()

    def test_send_recv_raw(self) -> None:
        coord_control_plane = ControlPlane.create(num_ranks=2, slot_size=512, rank=COORDINATOR)
        rank0_control_plane = ControlPlane.from_buffer(coord_control_plane.buf, rank=0, num_ranks=2, slot_size=512)
        try:
            seq1 = coord_control_plane.send(dst=0, payload=b"hello-rank-0", flags=FLAG_RAW)
            self.assertEqual(seq1, 1)

            message = rank0_control_plane.recv(src=COORDINATOR, timeout=1.0)
            self.assertEqual(message.seq, 1)
            self.assertEqual(message.payload, b"hello-rank-0")
            self.assertEqual(message.sender, COORDINATOR)
            self.assertEqual(message.destination, 0)

            # Reply back to coordinator
            seq2 = rank0_control_plane.send(dst=COORDINATOR, payload=b"reply-to-coord", flags=FLAG_RAW)
            self.assertEqual(seq2, 1)

            reply = coord_control_plane.recv(src=0, timeout=1.0)
            self.assertEqual(reply.seq, 1)
            self.assertEqual(reply.payload, b"reply-to-coord")
            self.assertEqual(reply.sender, 0)
        finally:
            coord_control_plane.close()

    def test_send_recv_json(self) -> None:
        coord_control_plane = ControlPlane.create(num_ranks=2, slot_size=1024, rank=COORDINATOR)
        rank1_control_plane = ControlPlane.from_buffer(coord_control_plane.buf, rank=1, num_ranks=2, slot_size=1024)
        try:
            data = {"step_id": 42, "tokens": [1, 2, 3], "layout": "b7"}
            coord_control_plane.send(dst=1, payload=data)

            message = rank1_control_plane.recv(src=COORDINATOR, timeout=1.0)
            self.assertEqual(message.flags, FLAG_JSON)
            self.assertEqual(message.json(), data)
        finally:
            coord_control_plane.close()

    def test_step_control_pack_unpack(self) -> None:
        step = StepControl(step_id=10, layout_id=128, draft_ids=(1, 2, 3), masks=(1, 1, 0))
        packed = step.pack()
        unpacked = StepControl.unpack(packed)
        self.assertEqual(unpacked, step)

    def test_acceptance_control_pack_unpack(self) -> None:
        acceptance = AcceptanceControl(step_id=10, accepted_len=3, accepted_tokens=(101, 102, 103))
        packed = acceptance.pack()
        unpacked = AcceptanceControl.unpack(packed)
        self.assertEqual(unpacked, acceptance)

    def test_rank_status_pack_unpack(self) -> None:
        status = RankStatus(step_id=10, rank=2, status_code=0)
        packed = status.pack()
        unpacked = RankStatus.unpack(packed)
        self.assertEqual(unpacked, status)

    def test_send_payload_exceeds_slot_size(self) -> None:
        control_plane = ControlPlane.create(num_ranks=2, slot_size=128)
        try:
            with self.assertRaises(ValueError):
                control_plane.send(dst=0, payload=b"x" * 256)
        finally:
            control_plane.close()

    def test_recv_timeout(self) -> None:
        control_plane = ControlPlane.create(num_ranks=2, slot_size=128, rank=0)
        try:
            with self.assertRaises(TimeoutError):
                control_plane.recv(src=COORDINATOR, timeout=0.01)
        finally:
            control_plane.close()

    def test_backpressure_and_missed_message(self) -> None:
        """Verifies sender backpressure and detection of missed sequence numbers."""
        coord = ControlPlane.create(num_ranks=2, slot_size=512, rank=COORDINATOR)
        rank0 = ControlPlane.from_buffer(coord.buf, rank=0, num_ranks=2, slot_size=512)
        try:
            # Send message 1
            coord.send(dst=0, payload=b"message-1")

            # Second send without recv should wait on backpressure and time out
            with self.assertRaises(TimeoutError):
                coord.send(dst=0, payload=b"message-2", timeout=0.02)

            # Receiver reads message 1, releasing backpressure
            msg1 = rank0.recv(src=COORDINATOR, timeout=0.1)
            self.assertEqual(msg1.payload, b"message-1")

            # Now sender can send message 2
            coord.send(dst=0, payload=b"message-2")
            msg2 = rank0.recv(src=COORDINATOR, timeout=0.1)
            self.assertEqual(msg2.payload, b"message-2")

            # Expecting an older sequence should raise RuntimeError (missed message)
            with self.assertRaises(RuntimeError):
                rank0.recv(src=COORDINATOR, expected_seq=1, timeout=0.01)
        finally:
            coord.close()

    def test_named_shm_create_attach_unlink(self) -> None:
        name = _unique_shm_name()
        control_plane = ControlPlane.create(name=name, num_ranks=4, slot_size=1024)
        try:
            attached = ControlPlane.attach(name=name, rank=0)
            self.assertEqual(attached.num_ranks, 4)
            self.assertEqual(attached.slot_size, 1024)
            attached.close()
        finally:
            control_plane.close()
            control_plane.unlink()

    def test_attach_does_not_unlink_on_process_exit(self) -> None:
        """Verifies that an independent attaching process exiting does not unlink the segment."""
        name = _unique_shm_name()
        control_plane = ControlPlane.create(name=name, num_ranks=2, slot_size=512)
        try:
            worker = mp.Process(target=_attaching_worker, args=(name,))
            worker.start()
            worker.join(timeout=5.0)
            self.assertEqual(worker.exitcode, 0)

            # The segment must still exist and be attachable after the worker process terminates
            reattached = ControlPlane.attach(name=name, rank=1)
            self.assertEqual(reattached.num_ranks, 2)
            reattached.close()
        finally:
            control_plane.close()
            control_plane.unlink()

    def test_rank_barrier_multiprocess(self) -> None:
        num_ranks = 4
        rounds = 50
        name = _unique_shm_name()
        control_plane = ControlPlane.create(name=name, num_ranks=num_ranks, slot_size=1024)
        try:
            procs = [
                mp.Process(target=_worker_rank_barrier, args=(r, num_ranks, rounds, name))
                for r in range(num_ranks)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=10.0)
                self.assertFalse(p.is_alive(), f"Rank process {p.pid} hung")
                self.assertEqual(p.exitcode, 0)
        finally:
            control_plane.close()
            control_plane.unlink()

    def test_all_barrier_multiprocess(self) -> None:
        num_ranks = 4
        total_participants = num_ranks + 1
        rounds = 50
        name = _unique_shm_name()
        control_plane = ControlPlane.create(name=name, num_ranks=num_ranks, slot_size=1024)
        try:
            procs = [
                mp.Process(target=_worker_all_barrier, args=(p, num_ranks, rounds, name))
                for p in range(total_participants)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=10.0)
                self.assertFalse(p.is_alive(), f"Participant process {p.pid} hung")
                self.assertEqual(p.exitcode, 0)
        finally:
            control_plane.close()
            control_plane.unlink()

    def test_fake_processes_step_round_trip_benchmark(self) -> None:
        """Measures full step control round trip across coordinator and 4 ranks."""
        num_ranks = 4
        step_count = 1000
        name = _unique_shm_name()
        control_plane = ControlPlane.create(name=name, num_ranks=num_ranks, slot_size=4096, rank=COORDINATOR)
        try:
            procs = [
                mp.Process(target=_fake_rank_step_loop, args=(r, num_ranks, step_count, name))
                for r in range(num_ranks)
            ]
            for p in procs:
                p.start()

            # Warmup
            time.sleep(0.05)
            t0 = time.perf_counter()
            for step in range(1, step_count + 1):
                # Coordinator publishes step message
                control_plane.publish_step(step_id=step, layout_id=128, draft_ids=(101, 102, 103), masks=(1, 1, 1))

                # Coordinator gathers acks from all ranks
                for r in range(num_ranks):
                    status = control_plane.read_status(rank=r, expected_step_id=step, timeout=5.0)
                    self.assertEqual(status.status_code, 0)
            t1 = time.perf_counter()

            for p in procs:
                p.join(timeout=10.0)
                self.assertFalse(p.is_alive())
                self.assertEqual(p.exitcode, 0)

            round_trip_us = (t1 - t0) / step_count * 1e6
            print(f"[vxpu] control plane round trip: {round_trip_us:.2f} us per step across coordinator and {num_ranks} ranks")
            if os.environ.get("VXPU_BENCHMARK") == "1":
                self.assertLess(round_trip_us, 50.0, f"Round trip time {round_trip_us:.2f} us exceeded target of 50 us")
        finally:
            control_plane.close()
            control_plane.unlink()


if __name__ == "__main__":
    unittest.main()
