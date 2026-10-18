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

"""Rehydrate a vXPU artifact into runnable modules.

bound tensors are fetched by HTTP range requests coalesced across adjacent
ranges, streaming straight into destination tensor storage (flat memory profile;
warm reloads come from the server's digest keep-alive). derived tensors are
recomputed from config using transformers' own formula registry (exact RoPE for
every rope_type, including per-layer-type hybrids). state tensors are
zero-initialized per session.

The graphs were traced on the meta device; after placement they are retargeted
to the executor's device, and device-specific compat passes are applied (e.g.
histc(int) has no CPU kernel; for expert counting it is exactly bincount).
"""

import collections
import fcntl
import hashlib
import math
import os
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from torch.export.passes import move_to_device_pass

from . import kernels

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}


_local = threading.local()


def _session():
    """One keep-alive HTTP session per thread.

    A model is tens of thousands of range reads; a new connection per
    read exhausts the ephemeral ports (errno 99) on a fast node.
    """
    session = getattr(_local, "session", None)
    if session is None:
        session = _local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


GAP_BYTES = 1 << 20          # 1 MB gap coalescence
RUN_PART_BYTES = 256 << 20   # split runs > 256 MB into parts for parallelism
CHUNK = 8 << 20


def _read_range(url, offset, length, scatter_list):
    """Stream bytes [offset, offset + length) of url demultiplexed into
    the destination scatter list `scatter_list` [(dest_tensor, dest_offset, seg_len), ...].
    Skips gap segments where dest_tensor is None."""
    end = offset + length - 1
    if url.startswith("file://"):  # a local checkpoint directory
        file_path = url[len("file://"):]
        with open(file_path, "rb") as f:
            f.seek(offset)
            for tensor, dest_offset, seg_len in scatter_list:
                if seg_len == 0:
                    continue
                if tensor is not None:
                    mv = memoryview(tensor.contiguous().view(torch.uint8).reshape(-1).numpy())
                    got = f.readinto(mv[dest_offset : dest_offset + seg_len])
                    if got != seg_len:
                        raise RuntimeError(f"expected {seg_len} bytes from {file_path}, got {got}")
                else:
                    f.seek(seg_len, os.SEEK_CUR)
        return

    resp = _session().get(url, headers={"Range": f"bytes={offset}-{end}"}, stream=True)
    resp.raise_for_status()
    if resp.status_code != 206:
        raise RuntimeError(f"expected HTTP 206 Partial Content, got {resp.status_code}")

    seg_idx = 0
    seg_offset = 0
    total_got = 0

    for chunk in resp.iter_content(CHUNK):
        chunk_view = memoryview(chunk)
        chunk_pos = 0
        chunk_len = len(chunk_view)
        total_got += chunk_len
        if total_got > length:
            raise RuntimeError(f"server sent more than {length} bytes")

        while chunk_pos < chunk_len:
            if seg_idx >= len(scatter_list):
                raise RuntimeError(f"server sent more than {length} bytes")
            tensor, dest_offset, seg_len = scatter_list[seg_idx]
            rem_seg = seg_len - seg_offset
            take = min(chunk_len - chunk_pos, rem_seg)
            if tensor is not None:
                mv = memoryview(tensor.contiguous().view(torch.uint8).reshape(-1).numpy())
                mv[dest_offset + seg_offset : dest_offset + seg_offset + take] = chunk_view[chunk_pos : chunk_pos + take]
            seg_offset += take
            chunk_pos += take
            if seg_offset == seg_len:
                seg_idx += 1
                seg_offset = 0

    if total_got != length or seg_idx != len(scatter_list):
        raise RuntimeError(f"expected {length} bytes, got {total_got} bytes")


_orig_read_range = _read_range


def _clean_strays(dirname, basename, exclude=None):
    try:
        entries = os.listdir(dirname)
    except OSError:
        return
    for name in entries:
        if name == basename + ".partial" or name.startswith(basename + ".partial."):
            file_path = os.path.join(dirname, name)
            if exclude and os.path.abspath(file_path) == os.path.abspath(exclude):
                continue
            try:
                sfd = os.open(file_path, os.O_RDWR)
                try:
                    fcntl.flock(sfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.remove(file_path)
                except (BlockingIOError, OSError):
                    pass
                finally:
                    os.close(sfd)
            except OSError:
                try:
                    os.remove(file_path)
                except OSError:
                    pass


class FileBacked(tuple):
    """(tensor, fresh, commit) tuple for memory-mapped storage.

    When `fresh` is True, storage is created under an exclusive temporary name
    and `commit()` atomically fsyncs and renames it to `path`. Can also be used
    as a context manager that commits on exit if no exception occurred.
    """
    def __new__(cls, tensor, fresh, commit):
        return super().__new__(cls, (tensor, fresh, commit))

    @property
    def tensor(self):
        return self[0]

    @property
    def fresh(self):
        return self[1]

    @property
    def commit(self):
        return self[2]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()


def file_backed(path, length):
    """A uint8 tensor of `length` bytes whose storage is the file at
    `path`, memory-mapped shared: writes land in the page cache and reach
    the file, reads come from the page cache or the disk. A tensor placed
    this way costs no committed memory; the kernel keeps the hot pages
    resident and evicts the rest, which is the RAM-over-SSD tier of
    hierarchical placement. A complete file of the right size is reused.

    When `fresh` is True, storage is created under an exclusive temporary
    name in the same directory; call `commit()` once the tensor is filled
    to fsync the data, atomically replace it to `path`, and fsync the
    directory.
    """
    dirname = os.path.dirname(path) or "."
    basename = os.path.basename(path)
    os.makedirs(dirname, exist_ok=True)
    fresh = not (os.path.exists(path) and os.path.getsize(path) == length)
    if not fresh:
        _clean_strays(dirname, basename)
        tensor = torch.from_file(path, shared=True, size=length, dtype=torch.uint8)
        tensor.commit = lambda: None
        return FileBacked(tensor, False, lambda: None)

    _clean_strays(dirname, basename)
    fd, tmp = tempfile.mkstemp(dir=dirname, prefix=basename + ".partial.")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        pass
    os.ftruncate(fd, length)
    tensor = torch.from_file(tmp, shared=True, size=length, dtype=torch.uint8)
    fin = weakref.finalize(tensor, os.close, fd)

    def commit():
        if os.path.exists(tmp):
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                fin()
            os.replace(tmp, path)
            dir_fd = os.open(dirname, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
            _clean_strays(dirname, basename)

    tensor.commit = commit
    return FileBacked(tensor, True, commit)


class CompletionTracker:
    """Thread-safe tracker for multiple tasks completing a shared destination."""
    def __init__(self, count, on_complete):
        self.remaining = count
        self.on_complete = on_complete
        self.lock = threading.Lock()

    def complete(self):
        with self.lock:
            self.remaining -= 1
            if self.remaining == 0:
                cb = self.on_complete
                self.on_complete = None
                if cb is not None:
                    cb()


class ReadItem:
    """A tensor slice to be read from a file range into a destination tensor."""

    def __init__(self, file_sha256, offset, length, target,
                 cat_dim=None, slot_offset=0, shape=None, dtype=None, cas_path=None,
                 on_complete=None):
        self.file_sha256 = file_sha256
        self.offset = offset
        self.length = length
        self.target = target
        self.cat_dim = cat_dim
        self.slot_offset = slot_offset
        self.shape = shape if shape is not None else (list(target.shape) if target is not None else None)
        self.dtype = dtype if dtype is not None else (target.dtype if target is not None else None)
        self.cas_path = cas_path
        self.on_complete = on_complete
        self.duplicates = []
        self.tracker = None

    def add_duplicate(self, item):
        self.duplicates.append(item)

    def is_contiguous_target(self):
        if self.length == 0 or self.target is None:
            return True
        if self.cat_dim is None:
            return self.target.is_contiguous()
        slice_tensor = self.target.narrow(self.cat_dim, self.slot_offset, self.shape[self.cat_dim])
        return slice_tensor.is_contiguous()

    def get_contiguous_target_and_offset(self):
        """Returns (tensor, offset) if contiguous, or (None, 0) if non-contiguous."""
        if self.length == 0 or self.target is None:
            return None, 0
        if self.cat_dim is None:
            if self.target.is_contiguous():
                t_u8 = self.target.contiguous().view(torch.uint8).reshape(-1)
                return t_u8, 0
            return None, 0
        slice_tensor = self.target.narrow(self.cat_dim, self.slot_offset, self.shape[self.cat_dim])
        if slice_tensor.is_contiguous():
            t_u8 = slice_tensor.contiguous().view(torch.uint8).reshape(-1)
            return t_u8, 0
        return None, 0


def _complete_item(item):
    """Finalize an item after its bytes have landed, notify callbacks, and drop host tensor references."""
    for dup in item.duplicates:
        if dup.target is not None and item.target is not None:
            d_dst = dup.target if dup.cat_dim is None else dup.target.narrow(dup.cat_dim, dup.slot_offset, dup.shape[dup.cat_dim])
            s_src = item.target if item.cat_dim is None else item.target.narrow(item.cat_dim, item.slot_offset, item.shape[item.cat_dim])
            d_dst.copy_(s_src)
        if dup.on_complete is not None:
            dup.on_complete()
        dup.target = None
        dup.on_complete = None
    del item.duplicates[:]

    if item.cas_path is not None and item.target is not None:
        os.makedirs(os.path.dirname(item.cas_path), exist_ok=True)
        tmp_cas = f"{item.cas_path}.partial"
        with open(tmp_cas, "wb") as f:
            t = item.target if item.cat_dim is None else item.target.narrow(item.cat_dim, item.slot_offset, item.shape[item.cat_dim])
            f.write(memoryview(t.contiguous().view(torch.uint8).reshape(-1).numpy()))
        os.replace(tmp_cas, item.cas_path)

    cb = item.on_complete
    item.on_complete = None
    if cb is not None:
        cb()

    item.target = None


def _read_cas_item(path, item):
    """Read a cached item from disk into its destination, and complete it."""
    if item.is_contiguous_target():
        t = item.target if item.cat_dim is None else item.target.narrow(item.cat_dim, item.slot_offset, item.shape[item.cat_dim])
        mv = memoryview(t.contiguous().view(torch.uint8).reshape(-1).numpy())
        with open(path, "rb") as f:
            if f.readinto(mv) != item.length:
                raise RuntimeError(f"cache entry {path} is short")
    else:
        tmp = torch.empty(item.shape, dtype=item.dtype)
        mv = memoryview(tmp.contiguous().view(torch.uint8).reshape(-1).numpy())
        with open(path, "rb") as f:
            if f.readinto(mv) != item.length:
                raise RuntimeError(f"cache entry {path} is short")
        dest_slice = item.target.narrow(item.cat_dim, item.slot_offset, item.shape[item.cat_dim])
        dest_slice.copy_(tmp)
        tmp = None

    _complete_item(item)


class _Run:
    """A contiguous or coalesced run of bytes to fetch from one file."""

    def __init__(self, file_sha256, offset, length, items):
        self.file_sha256 = file_sha256
        self.offset = offset
        self.length = length
        self.items = items


def build_part_spec(run, p_start, p_len):
    """Construct segment metadata (item, dest_off, seg_len, is_gap) for a run part without allocating tensors."""
    p_end = p_start + p_len
    part_spec = []
    touched = []
    cur_pos = p_start
    prev_end = None

    for item in run.items:
        i_start = item.offset
        i_end = item.offset + item.length
        if prev_end is not None:
            assert i_start >= prev_end, f"overlapping items detected in run: {i_start} < {prev_end}"
        prev_end = i_end

        if i_end <= p_start or i_start >= p_end:
            continue
        touched.append(item)
        if i_start > cur_pos:
            skip = min(i_start, p_end) - cur_pos
            part_spec.append((None, 0, skip, True))
            cur_pos += skip
        seg_start = max(i_start, cur_pos)
        seg_end = min(i_end, p_end)
        seg_len = seg_end - seg_start
        if seg_len > 0:
            dest_off = seg_start - i_start
            part_spec.append((item, dest_off, seg_len, False))
            cur_pos += seg_len
    if cur_pos < p_end:
        part_spec.append((None, 0, p_end - cur_pos, True))
    return part_spec, touched


def plan_and_fetch_ranges(ranges, files, cas_dir=None, workers=16,
                          gap_bytes=GAP_BYTES, part_bytes=RUN_PART_BYTES,
                          batch_size=64):
    """Plan, coalesce, and fetch ranges straight into destination tensors.

    Zero-length items are completed immediately at planning time without requests.
    Uncached items are grouped by file and merged across small gaps (gap_bytes)
    into large runs, then fetched in batched calls to fetch_ranges.
    Staging buffers for non-contiguous slices are allocated strictly at fetch time
    and freed immediately after copying.
    Network requests are dispatched before CAS hits so network latency is minimized.
    Host references to completed CPU tensors are dropped immediately upon completion.

    Returns (bytes_downloaded, requests_issued).
    """
    needed_items = []
    cas_hits = []

    for item in ranges:
        if item.length == 0:
            _complete_item(item)
            continue

        if cas_dir:
            key = hashlib.sha256(f"{item.file_sha256}:{item.offset}:{item.length}".encode()).hexdigest()
            path = os.path.join(cas_dir, key)
            if os.path.exists(path) and os.path.getsize(path) == item.length:
                cas_hits.append((path, item))
                continue
            item.cas_path = path
        needed_items.append(item)

    items_by_file = collections.defaultdict(list)
    for it in needed_items:
        items_by_file[it.file_sha256].append(it)

    runs = []
    for file_sha256, file_items in items_by_file.items():
        file_items.sort(key=lambda it: (it.offset, -it.length))
        cur_run = None
        for it in file_items:
            if cur_run is None:
                cur_run = _Run(file_sha256, it.offset, it.length, [it])
            else:
                cur_end = cur_run.offset + cur_run.length
                last_it = cur_run.items[-1]
                if it.offset < cur_end:
                    if it.offset == last_it.offset and it.length == last_it.length:
                        last_it.add_duplicate(it)
                        continue
                    raise ValueError(
                        f"overlapping items are not supported in coalesced runs: "
                        f"[{it.offset}, {it.offset + it.length}) overlaps [{last_it.offset}, {cur_end}) "
                        f"in file {file_sha256}"
                    )
                if it.offset <= cur_end + gap_bytes:
                    new_end = max(cur_end, it.offset + it.length)
                    cur_run.length = new_end - cur_run.offset
                    cur_run.items.append(it)
                else:
                    runs.append(cur_run)
                    cur_run = _Run(file_sha256, it.offset, it.length, [it])
        if cur_run is not None:
            runs.append(cur_run)

    tasks = []
    item_parts_count = collections.defaultdict(int)
    for run in runs:
        url = files[run.file_sha256]["source"]
        for p_off in range(0, run.length, part_bytes):
            p_len = min(part_bytes, run.length - p_off)
            p_start = run.offset + p_off
            part_spec, touched = build_part_spec(run, p_start, p_len)
            for it in touched:
                item_parts_count[it] += 1
            tasks.append((url, p_start, p_len, part_spec, touched))

    # Create thread-safe CompletionTrackers for all needed items
    for it, count in item_parts_count.items():
        it.tracker = CompletionTracker(count, lambda item=it: _complete_item(item))

    total_downloaded = sum(t[2] for t in tasks)
    requests_issued = len(tasks)

    # 1. Fetch network misses in batches to maximize connection pooling and native concurrency
    for b_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[b_start : b_start + batch_size]
        batch_requests = []
        batch_stagings = []

        # Allocate staging buffers strictly at fetch time
        for url, p_start, p_len, part_spec, touched in batch_tasks:
            scatter_list = []
            for it, dest_off, s_len, is_gap in part_spec:
                if is_gap:
                    scatter_list.append((None, 0, s_len))
                elif it.is_contiguous_target():
                    t_u8, base_off = it.get_contiguous_target_and_offset()
                    scatter_list.append((t_u8, base_off + dest_off, s_len))
                else:
                    # Allocate staging buffer AT FETCH TIME
                    tmp = torch.empty(s_len, dtype=torch.uint8)
                    scatter_list.append((tmp, 0, s_len))
                    batch_stagings.append((tmp, it, dest_off, s_len))
            batch_requests.append((url, p_start, p_len, scatter_list))

        # Perform the batched fetch (native fetch_ranges or fallback)
        if kernels.has_fetch() and _read_range is _orig_read_range:
            kernels.load_fetch().fetch_ranges(batch_requests, max_concurrent=batch_size)
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch_requests), workers)) as pool:
                futures = [pool.submit(_read_range, u, o, l, s) for u, o, l, s in batch_requests]
                for f in futures:
                    f.result()

        # Copy non-contiguous staging buffers into targets and immediately free staging memory
        for tmp, it, dest_off, s_len in batch_stagings:
            dest_slice = it.target.narrow(it.cat_dim, it.slot_offset, it.shape[it.cat_dim])
            part = tmp.view(it.dtype).view(it.shape)
            dest_slice.copy_(part)
        del batch_stagings[:]
        del batch_requests[:]

        # Notify trackers for completed parts in this batch; completed items drop CPU references immediately
        for url, p_start, p_len, part_spec, touched in batch_tasks:
            for it in touched:
                it.tracker.complete()

    # 2. Serve CAS hits concurrently in the thread pool (without delaying network requests)
    if cas_hits:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_read_cas_item, p, it) for p, it in cas_hits]
            for f in futures:
                f.result()

    return total_downloaded, requests_issued


def fetch_tensor(ref, files, cas_dir=None, mmap_dir=None):
    """One tensor, read by range into a destination tensor.

    Large tensors are read as concurrent ranges. With cas_dir set, bytes
    are also kept on disk under a content-derived key; warm reloads otherwise
    come from the engine keep-alive. With mmap_dir set, the buffer itself is a
    file there (see file_backed): the tensor is memory-mapped rather than held,
    and a file already complete from an earlier load is used without fetching.

    Returns (tensor, bytes_downloaded); 0 when served from the cache.
    """
    length = ref["length"]
    dtype = DTYPES[ref["dtype"]]
    shape = ref["shape"]
    if mmap_dir:
        key = hashlib.sha256(f"{ref['file_sha256']}:{ref['offset']}:{length}".encode()).hexdigest()
        fb = file_backed(os.path.join(mmap_dir, key), length)
        tensor = fb.tensor.view(dtype).view(shape)
        if not fb.fresh:
            return tensor, 0
        commit = fb.commit
    else:
        tensor = torch.empty(shape, dtype=dtype)
        commit = None

    item = ReadItem(ref["file_sha256"], ref["offset"], length, tensor,
                    shape=shape, dtype=dtype, on_complete=commit)
    downloaded, _ = plan_and_fetch_ranges([item], files, cas_dir=cas_dir)
    return tensor, downloaded


_orig_fetch_tensor = fetch_tensor


def fetch_stacked(spec, manifest_tensors, files, cas_dir=None, workers=16, mmap_path=None):
    """Assemble a parameter from several checkpoint tensors.

    ``spec["parts"]`` is a list of rows; each row is a list of manifest
    tensor names concatenated along ``spec["cat_dim"]`` (a gated MLP
    expert's gate and up projections become one block). With
    ``spec["stack"]`` (the default) the rows are stacked along a new
    leading dimension (an expert bank); otherwise there is one row (a
    table stored in shards). The result is allocated once and filled
    part by part, so assembling a 50 GB table costs 50 GB, not twice
    that. Parts are fetched concurrently.

    Returns (tensor, bytes_downloaded).
    """
    rows, cat_dim, stack = spec["parts"], spec["cat_dim"], spec.get("stack", True)
    first = manifest_tensors[rows[0][0]]
    row_shape = list(first["shape"])
    row_shape[cat_dim] = sum(manifest_tensors[n]["shape"][cat_dim] for n in rows[0])
    out_shape = ([len(rows)] if stack else []) + row_shape
    dtype = DTYPES[first["dtype"]]
    if mmap_path:
        nbytes = torch.empty(0, dtype=dtype).element_size()
        for d in out_shape:
            nbytes *= d
        fb = file_backed(mmap_path, nbytes)
        out = fb.tensor.view(dtype).view(out_shape)
        if not fb.fresh:
            return out, 0
        commit = fb.commit
    else:
        out = torch.empty(out_shape, dtype=dtype)
        commit = None

    if fetch_tensor is not _orig_fetch_tensor:
        downloaded = 0
        for r, row in enumerate(rows):
            target_row = out[r] if stack else out
            offset = 0
            for n in row:
                ref = manifest_tensors[n]
                t, dl = fetch_tensor(ref, files, cas_dir, mmap_dir=None)
                downloaded += dl
                target_row.narrow(cat_dim, offset, t.shape[cat_dim]).copy_(t)
                offset += t.shape[cat_dim]
        if commit:
            commit()
        return out, downloaded

    items = []
    total_parts = sum(len(row) for row in rows)
    tracker = CompletionTracker(total_parts, commit) if commit else None

    for r, row in enumerate(rows):
        target_row = out[r] if stack else out
        offset = 0
        for n in row:
            ref = manifest_tensors[n]
            part_shape = ref["shape"]
            part_dtype = DTYPES[ref["dtype"]]
            items.append(ReadItem(
                file_sha256=ref["file_sha256"],
                offset=ref["offset"],
                length=ref["length"],
                target=target_row,
                cat_dim=cat_dim,
                slot_offset=offset,
                shape=part_shape,
                dtype=part_dtype,
                on_complete=tracker.complete if tracker else None,
            ))
            offset += part_shape[cat_dim]

    downloaded, _ = plan_and_fetch_ranges(items, files, cas_dir=cas_dir, workers=workers)
    return out, downloaded


def derived_tensor(fqn, meta, config):
    """Recompute a config-derived tensor to the shape the graph expects.

    RoPE tables use transformers' own ROPE_INIT_FUNCTIONS registry,
    keyed by rope_type and (for hybrid-attention models) layer type —
    exact by construction. Embedding scales are sqrt(hidden).
    """
    text_config = (config.get_text_config()
                   if hasattr(config, "get_text_config") else config)
    shape = tuple(meta.shape)
    dtype = (meta.dtype if meta.dtype.is_floating_point
             else torch.float32)

    if "embed_scale" in fqn:
        hidden = text_config.hidden_size
        if "per_layer" in fqn and getattr(
                text_config, "hidden_size_per_layer_input", 0):
            hidden = text_config.hidden_size_per_layer_input
        return torch.full(shape or (1,), math.sqrt(hidden),
                          dtype=dtype).reshape(shape)

    # Mirror transformers' own rope init (modeling_rope_utils + the
    # per-model rotary modules): select the layer's rope_type, call the
    # registered init fn, and pass the exact kwargs transformers passes
    # (e.g. head_dim_key="global_head_dim" for Gemma 4 proportional
    # full-attention RoPE). Reproducing this rather than re-deriving
    # formulas keeps hybrid/partial-rotary models exact.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    layer_type = None
    for candidate in ("sliding_attention", "full_attention"):
        if candidate in fqn:
            layer_type = candidate
    rope_params = getattr(text_config, "rope_parameters", None) or {}
    if layer_type and isinstance(rope_params.get(layer_type), dict):
        params = rope_params[layer_type]
    else:
        params = rope_params if isinstance(rope_params, dict) else {}
    rope_type = params.get("rope_type", "default")

    kwargs = {}
    if layer_type:
        kwargs["layer_type"] = layer_type
        if layer_type == "full_attention" and rope_type == "proportional":
            kwargs["head_dim_key"] = "global_head_dim"
    init_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
    if init_fn is not None:
        inv_freq, _ = init_fn(text_config, **kwargs)
        want = shape[-1] if shape else inv_freq.numel()
        if inv_freq.numel() == want:
            return inv_freq.to(dtype).reshape(shape)
    # Fallback: plain geometric series over the dims the graph expects.
    length = shape[-1] if shape else 32
    theta = params.get("rope_theta",
                       getattr(text_config, "rope_theta", 10000.0))
    exponent = torch.arange(0, length, dtype=torch.float32) / max(length, 1)
    return (1.0 / (theta ** exponent)).to(dtype).reshape(shape)


def cpu_compat(exported):
    """Rewrite ops whose CPU kernels lack integer support.

    histc(int) has no CPU kernel; for integer inputs counted over
    [0, bins) it is exactly bincount (sliced to bins, cast back).
    Graphs from the forked transformers count in float off the GPU and
    need no rewrite; this remains for artifacts exported before that.
    """
    graph = exported.graph_module.graph
    replaced = 0
    for node in list(graph.nodes):
        if (node.op == "call_function"
                and str(node.target) == "aten.histc.default"):
            x, bins = node.args[0], node.args[1]
            val = x.meta.get("val") if hasattr(x, "meta") else None
            if val is not None and val.is_floating_point():
                continue  # float histc has a CPU kernel; leave the graph alone
            with graph.inserting_before(node):
                flat = graph.call_function(
                    torch.ops.aten.reshape.default, (x, [-1]))
                cast = graph.call_function(
                    torch.ops.aten._to_copy.default, (flat,),
                    {"dtype": torch.int64})
                counts = graph.call_function(
                    torch.ops.aten.bincount.default, (cast,),
                    {"minlength": bins})
                sliced = graph.call_function(
                    torch.ops.aten.slice.Tensor, (counts, 0, 0, bins))
                out = graph.call_function(
                    torch.ops.aten._to_copy.default, (sliced,),
                    {"dtype": torch.int32})
            node.replace_all_uses_with(out)
            graph.erase_node(node)
            replaced += 1
    exported.graph_module.recompile()
    return replaced


def strip_asserts(module):
    """Remove _assert_tensor_metadata guard nodes.

    They are eager-mode sanity checks Dynamo cannot re-trace when
    compiling on CUDA, and ep.module() regenerates some in submodules —
    strip every graph in the tree.
    """
    removed = 0
    for _, sub in module.named_modules():
        if not hasattr(sub, "graph"):
            continue
        for node in list(sub.graph.nodes):
            if (node.op == "call_function"
                    and "_assert_tensor_metadata" in str(node.target)):
                sub.graph.erase_node(node)
                removed += 1
        sub.recompile()
    return removed


def load_program(graph_path, tensors, device):
    """Load a .pt2, place real tensors into it, retarget to device."""
    exported = torch.export.load(graph_path)

    def place(fqn, tensor):
        # The checkpoint's dtype may differ from the graph's parameter
        # dtype (Qwen stores fp8 block scales as bf16; transformers'
        # fp8 modules declare float32). Cast as from_pretrained would.
        expected = exported.state_dict.get(fqn, exported.constants.get(fqn))
        bytes_like = (torch.int8, torch.uint8)
        if (isinstance(expected, torch.Tensor) and expected.dtype != tensor.dtype
                and expected.dtype in bytes_like and tensor.dtype in bytes_like):
            # Packed fp4 weights: the checkpoint says int8, the graph
            # keeps them as uint8 bytes. Same bytes, no cast.
            tensor = tensor.view(expected.dtype)
        elif (isinstance(expected, torch.Tensor)
                and expected.dtype != tensor.dtype
                and expected.is_floating_point()
                and tensor.is_floating_point()):
            tensor = tensor.to(expected.dtype)
        if fqn in exported.state_dict:
            exported.state_dict[fqn] = torch.nn.Parameter(
                tensor, requires_grad=False)
        elif fqn in exported.constants:
            exported.constants[fqn] = tensor
        # A graph need not lift every artifact tensor (prefill vs
        # decode differ slightly); extras are simply unused here.

    for fqn, tensor in tensors.items():
        place(fqn, tensor)

    # Example inputs were recorded on meta; materialize placeholders so
    # the device pass can walk them.
    ex_args, ex_kwargs = exported.example_inputs
    fix = lambda t: (torch.zeros(t.shape, dtype=t.dtype)  # noqa: E731
                     if isinstance(t, torch.Tensor) and t.is_meta else t)
    exported._example_inputs = (
        tuple(fix(t) for t in ex_args),
        {k: fix(v) for k, v in ex_kwargs.items()})

    exported = move_to_device_pass(exported, device)
    if device == "cpu":
        cpu_compat(exported)
    return exported


def share_state(src_module, dst_module, state_fqns):
    """Alias state buffers so prefill and decode share one cache.

    move_to_device_pass copies tensors per program; re-point the decode
    module's state buffers at the prefill module's tensors.
    """
    src = dict(src_module.named_buffers())
    shared = 0
    for fqn in state_fqns:
        if fqn in src:
            parent, name = (fqn.rsplit(".", 1)
                            if "." in fqn else ("", fqn))
            dst_parent = (dst_module.get_submodule(parent)
                          if parent else dst_module)
            if name in dst_parent._buffers:
                dst_parent._buffers[name] = src[fqn]
                shared += 1
    return shared
