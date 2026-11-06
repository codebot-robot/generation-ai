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

"""The control plane in shared memory for process-per-rank execution (#539).

This library coordinates executor card processes and a coordinator across
process boundaries. It provides dedicated message slots, monotonic sequence
numbers, consumer acknowledgement backpressure, and barriers aligned to
CPU cache lines.

Each sender and destination pair owns a dedicated message slot. Readers poll
monotonic sequence numbers, avoiding kernel crossings during decode steps.
A consumer acknowledgement field prevents senders from overwriting unread
messages. Sequence re-checking and checksums prevent torn reads.
"""

from dataclasses import dataclass
import json
import mmap
from multiprocessing import shared_memory
import struct
import time
from typing import Any
import zlib

MAGIC = b"VXMB"
VERSION = 1
CACHELINE = 64
DEFAULT_SLOT_SIZE = 4096

COORDINATOR = -1
BROADCAST = -2

FLAG_RAW = 0
FLAG_JSON = 1
FLAG_STEP = 2
FLAG_ACCEPT = 3
FLAG_STATUS = 4


@dataclass(frozen=True)
class StepControl:
    """Binary step parameters published by the coordinator."""

    step_id: int
    layout_id: int
    draft_ids: tuple[int, ...] = ()
    masks: tuple[int, ...] = ()

    def pack(self) -> bytes:
        """Packs step parameters into little-endian binary bytes."""
        count = len(self.draft_ids)
        header = struct.pack("<QII", self.step_id, self.layout_id, count)
        draft_bytes = struct.pack(f"<{count}q", *self.draft_ids) if count else b""
        mask_bytes = bytes(self.masks) if self.masks else b""
        return header + draft_bytes + mask_bytes

    @classmethod
    def unpack(cls, buf: bytes) -> "StepControl":
        """Unpacks step parameters from binary bytes."""
        step_id, layout_id, count = struct.unpack_from("<QII", buf, 0)
        drafts = struct.unpack_from(f"<{count}q", buf, 16) if count else ()
        masks_offset = 16 + count * 8
        masks = tuple(buf[masks_offset : masks_offset + count]) if len(buf) >= masks_offset + count else ()
        return cls(step_id=step_id, layout_id=layout_id, draft_ids=drafts, masks=masks)


@dataclass(frozen=True)
class AcceptanceControl:
    """Binary acceptance choices published by rank 0."""

    step_id: int
    accepted_len: int
    accepted_tokens: tuple[int, ...] = ()

    def pack(self) -> bytes:
        """Packs acceptance choices into little-endian binary bytes."""
        count = len(self.accepted_tokens)
        header = struct.pack("<QII", self.step_id, self.accepted_len, count)
        token_bytes = struct.pack(f"<{count}q", *self.accepted_tokens) if count else b""
        return header + token_bytes

    @classmethod
    def unpack(cls, buf: bytes) -> "AcceptanceControl":
        """Unpacks acceptance choices from binary bytes."""
        step_id, accepted_len, count = struct.unpack_from("<QII", buf, 0)
        tokens = struct.unpack_from(f"<{count}q", buf, 16) if count else ()
        return cls(step_id=step_id, accepted_len=accepted_len, accepted_tokens=tokens)


@dataclass(frozen=True)
class RankStatus:
    """Binary status returned by an executor rank to the coordinator."""

    step_id: int
    rank: int
    status_code: int = 0

    def pack(self) -> bytes:
        """Packs rank status into little-endian binary bytes."""
        return struct.pack("<QII", self.step_id, self.rank, self.status_code)

    @classmethod
    def unpack(cls, buf: bytes) -> "RankStatus":
        """Unpacks rank status from binary bytes."""
        step_id, rank, status_code = struct.unpack_from("<QII", buf, 0)
        return cls(step_id=step_id, rank=rank, status_code=status_code)


@dataclass(frozen=True)
class Message:
    """A control message retrieved from a shared-memory slot."""

    seq: int
    payload: bytes
    flags: int
    sender: int
    destination: int

    @property
    def text(self) -> str:
        """Decodes the payload as a UTF-8 string."""
        return self.payload.decode("utf-8")

    def json(self) -> Any:
        """Parses the payload as JSON data."""
        return json.loads(self.text)

    def as_step(self) -> StepControl:
        """Unpacks binary step control data."""
        return StepControl.unpack(self.payload)

    def as_acceptance(self) -> AcceptanceControl:
        """Unpacks binary acceptance control data."""
        return AcceptanceControl.unpack(self.payload)

    def as_status(self) -> RankStatus:
        """Unpacks binary rank status data."""
        return RankStatus.unpack(self.payload)

    def __bytes__(self) -> bytes:
        return self.payload

    def __len__(self) -> int:
        return len(self.payload)


class ControlPlane:
    """The shared-memory control plane coordinating rank processes.

    The memory segment contains a 64-byte file header, barrier state arrays,
    and fixed-size message slots.
    """

    def __init__(self, buf: Any, rank: int, num_ranks: int, slot_size: int = DEFAULT_SLOT_SIZE,
                 shared_memory_resource: Any | None = None, is_owner: bool = False):
        if rank != COORDINATOR and not (0 <= rank < num_ranks):
            raise ValueError(f"Invalid rank {rank}: must be COORDINATOR ({COORDINATOR}) or in range 0..{num_ranks - 1}")

        self.buf = buf
        self.rank = rank
        self.num_ranks = num_ranks
        self.slot_size = slot_size
        self._shared_memory_resource = shared_memory_resource
        self._is_owner = is_owner

        self.num_senders = num_ranks + 1
        self.num_destinations = num_ranks + 2
        self.num_slots = self.num_senders * self.num_destinations

        self.header_size = 64
        self.rank_barrier_offset = self.header_size
        self.all_barrier_offset = self.rank_barrier_offset + num_ranks * CACHELINE

        barriers_end = self.all_barrier_offset + self.num_senders * CACHELINE
        self.slots_offset = (barriers_end + CACHELINE - 1) & ~(CACHELINE - 1)
        self.slot_stride = 64 + slot_size

        self._out_seq: dict[tuple[int, int], int] = {}
        self._in_seq: dict[tuple[int, int], int] = {}
        self._rank_barrier_round = 0
        self._all_barrier_round = 0

    @classmethod
    def total_bytes(cls, num_ranks: int, slot_size: int = DEFAULT_SLOT_SIZE) -> int:
        """Calculates total shared-memory bytes needed for N ranks and slot size."""
        num_senders = num_ranks + 1
        num_destinations = num_ranks + 2
        num_slots = num_senders * num_destinations
        barriers_end = 64 + num_ranks * CACHELINE + num_senders * CACHELINE
        slots_offset = (barriers_end + CACHELINE - 1) & ~(CACHELINE - 1)
        total = slots_offset + num_slots * (64 + slot_size)
        return (total + 4095) & ~4095

    @classmethod
    def create(cls, name: str | None = None, num_ranks: int = 4,
               slot_size: int = DEFAULT_SLOT_SIZE, rank: int = COORDINATOR) -> "ControlPlane":
        """Creates a new shared-memory control plane segment.

        When name is None, allocates anonymous shared memory suitable for
        forked processes. When name is provided, allocates a named POSIX
        shared-memory block that independent processes attach to.
        """
        size = cls.total_bytes(num_ranks, slot_size)
        if name is not None:
            shared_mem = shared_memory.SharedMemory(create=True, size=size, name=name)
            buf = shared_mem.buf
            resource = shared_mem
        else:
            buf = mmap.mmap(-1, size, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
            resource = buf

        # Zero header, barrier areas, and slot headers
        buf[0:size] = b"\x00" * size

        # Initialize header: magic (4B), version (4B), num_ranks (4B), slot_size (4B), size (8B)
        buf[0:4] = MAGIC
        struct.pack_into("<IIIQ", buf, 4, VERSION, num_ranks, slot_size, size)

        return cls(buf=buf, rank=rank, num_ranks=num_ranks, slot_size=slot_size,
                   shared_memory_resource=resource, is_owner=True)

    @classmethod
    def attach(cls, name: str, rank: int) -> "ControlPlane":
        """Attaches to an existing named shared-memory control plane segment."""
        try:
            shared_mem = shared_memory.SharedMemory(create=False, name=name, track=False)
        except TypeError:
            # Python 3.12 does not support track=False in SharedMemory constructor.
            shared_mem = shared_memory.SharedMemory(create=False, name=name)
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(shared_mem._name, "shared_memory")
            except Exception:
                pass

        buf = shared_mem.buf

        magic = bytes(buf[0:4])
        if magic != MAGIC:
            shared_mem.close()
            raise ValueError(f"Invalid shared memory magic: expected {MAGIC!r}, got {magic!r}")

        version, num_ranks, slot_size, _ = struct.unpack_from("<IIIQ", buf, 4)
        if version != VERSION:
            shared_mem.close()
            raise ValueError(f"Unsupported control plane version: expected {VERSION}, got {version}")

        return cls(buf=buf, rank=rank, num_ranks=num_ranks, slot_size=slot_size,
                   shared_memory_resource=shared_mem, is_owner=False)

    @classmethod
    def from_buffer(cls, buf: Any, rank: int, num_ranks: int,
                    slot_size: int = DEFAULT_SLOT_SIZE) -> "ControlPlane":
        """Wraps an existing buffer or memoryview directly."""
        return cls(buf=buf, rank=rank, num_ranks=num_ranks, slot_size=slot_size)

    def slot_offset(self, src: int, dst: int) -> int:
        """Computes the byte offset of the slot for a (src, dst) pair."""
        if src != COORDINATOR and not (0 <= src < self.num_ranks):
            raise ValueError(f"Invalid src {src}: must be COORDINATOR ({COORDINATOR}) or in range 0..{self.num_ranks - 1}")
        if dst != COORDINATOR and dst != BROADCAST and not (0 <= dst < self.num_ranks):
            raise ValueError(f"Invalid dst {dst}: must be COORDINATOR, BROADCAST, or in range 0..{self.num_ranks - 1}")

        sender_index = self.num_ranks if src == COORDINATOR else src
        destination_index = self.num_ranks if dst == COORDINATOR else (self.num_ranks + 1 if dst == BROADCAST else dst)
        slot_index = sender_index * self.num_destinations + destination_index
        return self.slots_offset + slot_index * self.slot_stride

    def _wait_value(self, offset: int, target_value: int, timeout: float | None = None) -> int:
        """Spins until the 64-bit integer at offset reaches target_value."""
        start_time = time.perf_counter()
        spin_count = 0
        while True:
            current_value = struct.unpack_from("<Q", self.buf, offset)[0]
            if current_value >= target_value:
                return current_value
            spin_count += 1
            if (spin_count & 1023) == 0:
                if timeout is not None and (time.perf_counter() - start_time) > timeout:
                    raise TimeoutError(f"Wait timed out at offset {offset}: expected {target_value}, got {current_value}")
                if spin_count > 50000:
                    time.sleep(0.0001)

    def send(self, dst: int, payload: bytes | str | dict | list | StepControl | AcceptanceControl | RankStatus,
             flags: int = FLAG_RAW, seq: int | None = None, timeout: float | None = None) -> int:
        """Sends a control message to destination dst.

        Encodes data into bytes, enforces consumer backpressure on previous messages,
        writes payload and header checksum, then commits the sequence number.
        """
        if isinstance(payload, StepControl):
            data = payload.pack()
            flags = FLAG_STEP
        elif isinstance(payload, AcceptanceControl):
            data = payload.pack()
            flags = FLAG_ACCEPT
        elif isinstance(payload, RankStatus):
            data = payload.pack()
            flags = FLAG_STATUS
        elif isinstance(payload, (dict, list)):
            data = json.dumps(payload).encode("utf-8")
            if flags == FLAG_RAW:
                flags = FLAG_JSON
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            data = bytes(payload)
        else:
            raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

        if len(data) > self.slot_size:
            raise ValueError(f"Payload size {len(data)} exceeds slot size {self.slot_size}")

        offset = self.slot_offset(self.rank, dst)
        slot_key = (self.rank, dst)

        # Enforce backpressure for unicast destinations: ensure previous message was acknowledged.
        if dst != BROADCAST:
            prev_seq = self._out_seq.get(slot_key, 0)
            if prev_seq > 0:
                self._wait_value(offset + 32, prev_seq, timeout=timeout)

        if seq is None:
            self._out_seq[slot_key] = self._out_seq.get(slot_key, 0) + 1
            seq = self._out_seq[slot_key]
        else:
            self._out_seq[slot_key] = seq

        count = len(data)
        checksum = zlib.crc32(data, seq & 0xFFFFFFFF)

        # Write payload first
        self.buf[offset + 64 : offset + 64 + count] = data
        # Write metadata header at offset + 8: length (4B), flags (4B), sender (4B), destination (4B), crc (4B)
        struct.pack_into("<IIiiI", self.buf, offset + 8, count, flags, self.rank, dst, checksum)
        # Commit sequence number at offset + 0 to publish message
        struct.pack_into("<Q", self.buf, offset, seq)
        return seq

    def recv(self, src: int, dst: int | None = None,
             expected_seq: int | None = None, timeout: float | None = None) -> Message:
        """Waits for and reads a control message from sender src.

        Spins for low latency on fast steps, verifying sequence numbers and checksums
        to prevent torn reads. Acknowledges receipt for unicast backpressure.
        """
        if dst is None:
            dst = self.rank
        offset = self.slot_offset(src, dst)
        slot_key = (src, dst)
        if expected_seq is None:
            expected_seq = self._in_seq.get(slot_key, 0) + 1

        start_time = time.perf_counter()
        spin_count = 0
        while True:
            cur_seq = struct.unpack_from("<Q", self.buf, offset)[0]
            if cur_seq < expected_seq:
                spin_count += 1
                if (spin_count & 1023) == 0:
                    if timeout is not None and (time.perf_counter() - start_time) > timeout:
                        raise TimeoutError(f"recv timed out waiting for seq {expected_seq} from {src} (slot {slot_key})")
                    if spin_count > 50000:
                        time.sleep(0.0001)
                continue

            if cur_seq > expected_seq:
                raise RuntimeError(
                    f"Missed message from {src} to {dst}: expected sequence {expected_seq}, slot already at {cur_seq}"
                )

            # Sequence matches expected_seq. Read header and payload.
            count, flags, sender, destination, expected_checksum = struct.unpack_from("<IIiiI", self.buf, offset + 8)
            payload = bytes(self.buf[offset + 64 : offset + 64 + count])

            # Re-check sequence to detect torn read from concurrent sender write.
            post_seq = struct.unpack_from("<Q", self.buf, offset)[0]
            if post_seq != cur_seq:
                continue

            # Verify checksum against payload seeded with sequence.
            if zlib.crc32(payload, cur_seq & 0xFFFFFFFF) != expected_checksum:
                continue

            # Acknowledge receipt for unicast destinations to release sender backpressure.
            if dst != BROADCAST:
                struct.pack_into("<Q", self.buf, offset + 32, cur_seq)

            self._in_seq[slot_key] = cur_seq
            return Message(seq=cur_seq, payload=payload, flags=flags, sender=sender, destination=destination)

    def _sync_barrier(self, base_offset: int, participant_count: int,
                      participant_index: int, round_number: int,
                      timeout: float | None = None) -> None:
        """Synchronizes participants using a single-phase monotonic round counter."""
        struct.pack_into("<Q", self.buf, base_offset + participant_index * CACHELINE, round_number)
        for i in range(participant_count):
            self._wait_value(base_offset + i * CACHELINE, round_number, timeout=timeout)

    def rank_barrier(self, timeout: float | None = None) -> None:
        """Synchronizes card ranks (0 through num_ranks-1) across a step boundary."""
        if self.rank < 0 or self.rank >= self.num_ranks:
            raise ValueError(f"Participant rank {self.rank} is not a worker rank (0..{self.num_ranks - 1})")
        self._rank_barrier_round += 1
        self._sync_barrier(self.rank_barrier_offset, self.num_ranks,
                           self.rank, self._rank_barrier_round, timeout)

    def barrier(self, timeout: float | None = None) -> None:
        """Synchronizes all participants, including the coordinator and ranks."""
        self._all_barrier_round += 1
        participant_index = self.num_ranks if self.rank == COORDINATOR else self.rank
        self._sync_barrier(self.all_barrier_offset, self.num_senders,
                           participant_index, self._all_barrier_round, timeout)

    # ---- High-level control message helpers ----

    def publish_step(self, step_id: int, layout_id: int = 0,
                     draft_ids: tuple[int, ...] | list[int] | None = None,
                     masks: tuple[int, ...] | list[int] | None = None) -> int:
        """Publishes step parameters from the coordinator to all ranks."""
        control = StepControl(
            step_id=step_id,
            layout_id=layout_id,
            draft_ids=tuple(draft_ids) if draft_ids else (),
            masks=tuple(masks) if masks else ()
        )
        return self.send(dst=BROADCAST, payload=control, seq=step_id)

    def read_step(self, expected_step_id: int | None = None,
                  timeout: float | None = None) -> StepControl:
        """Reads the step broadcast published by the coordinator."""
        message = self.recv(src=COORDINATOR, dst=BROADCAST, expected_seq=expected_step_id, timeout=timeout)
        return message.as_step()

    def publish_acceptance(self, step_id: int, accepted_len: int,
                           accepted_tokens: tuple[int, ...] | list[int] | None = None) -> int:
        """Publishes acceptance results from rank 0 to peer ranks."""
        control = AcceptanceControl(
            step_id=step_id,
            accepted_len=accepted_len,
            accepted_tokens=tuple(accepted_tokens) if accepted_tokens else ()
        )
        return self.send(dst=BROADCAST, payload=control, seq=step_id)

    def read_acceptance(self, expected_step_id: int | None = None,
                        timeout: float | None = None) -> AcceptanceControl:
        """Reads acceptance results published by rank 0."""
        message = self.recv(src=0, dst=BROADCAST, expected_seq=expected_step_id, timeout=timeout)
        return message.as_acceptance()

    def post_status(self, step_id: int, status_code: int = 0) -> int:
        """Posts rank execution status back to the coordinator."""
        status = RankStatus(step_id=step_id, rank=self.rank, status_code=status_code)
        return self.send(dst=COORDINATOR, payload=status, seq=step_id)

    def read_status(self, rank: int, expected_step_id: int | None = None,
                    timeout: float | None = None) -> RankStatus:
        """Reads status from an executor rank at the coordinator."""
        message = self.recv(src=rank, dst=COORDINATOR, expected_seq=expected_step_id, timeout=timeout)
        return message.as_status()

    # ---- Resource cleanup ----

    def close(self) -> None:
        """Closes the shared-memory buffer or mapping."""
        if hasattr(self.buf, "close"):
            self.buf.close()
        elif self._shared_memory_resource is not None and hasattr(self._shared_memory_resource, "close"):
            self._shared_memory_resource.close()

    def unlink(self) -> None:
        """Unlinks named shared memory if this instance owns the allocation."""
        if self._is_owner and self._shared_memory_resource is not None and hasattr(self._shared_memory_resource, "unlink"):
            try:
                self._shared_memory_resource.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "ControlPlane":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
        if self._is_owner:
            self.unlink()
