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

"""Content-addressed model manifests, built from Hub metadata only.

A manifest describes a model without containing it: the architecture
config plus a table binding every tensor to
(file_sha256, byte_offset, byte_length, dtype, shape). It is built from
kilobytes of metadata — per-file sha256 (computed at upload, served by
the Hub API) and range-fetched safetensors headers — never from weight
bytes. The sha256 of the manifest is the model's digest.
"""

import hashlib
import json
import os
import struct

import requests
from huggingface_hub import HfApi, hf_hub_download

MANIFEST_FORMAT = "vxpu-manifest/v1alpha1"


def _fetch_range(url, start, length):
    if url.startswith("file://"):
        with open(url[len("file://"):], "rb") as f:
            f.seek(start)
            content = f.read(length)
        if len(content) != length:
            raise RuntimeError(
                f"expected {length} bytes from {url}, got {len(content)} bytes")
        return content
    resp = requests.get(
        url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    resp.raise_for_status()
    if resp.status_code != 206:
        raise RuntimeError(
            f"expected HTTP 206 Partial Content, got {resp.status_code}")
    content = resp.content
    if len(content) != length:
        raise RuntimeError(
            f"expected {length} bytes from {url}, got {len(content)} bytes")
    return content


def _local_shards(directory):
    """(sha256, size, name, file:// source) for each safetensors file in
    a local checkpoint directory (a save_pretrained output)."""
    shards = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(directory, name)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        shards.append((digest.hexdigest(), os.path.getsize(path), name,
                       "file://" + os.path.abspath(path)))
    return shards


def build_manifest(repo_id, revision="main"):
    """Build a manifest for a Hugging Face model repo, or for a local
    checkpoint directory (file:// sources; sha256 computed here).

    For a Hub repo this downloads no weight bytes: only file metadata
    and safetensors headers (a few KB per shard, via HTTP range
    requests).
    """
    if os.path.isdir(repo_id):
        directory = os.path.abspath(repo_id)
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        shards = _local_shards(directory)
        commit_sha, revision = "local", "local"
        repo_id = directory
    else:
        info = HfApi().model_info(repo_id, revision=revision,
                                  files_metadata=True)
        commit_sha = getattr(info, "sha", None) or revision
        with open(hf_hub_download(repo_id, "config.json",
                                  revision=commit_sha)) as f:
            config = json.load(f)
        shards = [(s.lfs.sha256, s.size, s.rfilename,
                   f"https://huggingface.co/{repo_id}/resolve/"
                   f"{commit_sha}/{s.rfilename}")
                  for s in info.siblings if s.rfilename.endswith(".safetensors")]

    files, tensors = {}, {}
    for sha, size, name, source in shards:
        files[sha] = {"size": size, "name": name, "source": source}
        (header_len,) = struct.unpack("<Q", _fetch_range(source, 0, 8))
        header = json.loads(_fetch_range(source, 8, header_len))
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            tensors[name] = {
                "file_sha256": sha,
                "offset": data_start + begin,
                "length": end - begin,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
            }

    return {
        "format": MANIFEST_FORMAT,
        "source": {"repo": repo_id, "revision": revision, "commit_sha": commit_sha},
        "config": config,
        "files": files,
        "tensors": tensors,
    }
