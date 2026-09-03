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

import json
import struct

import requests
from huggingface_hub import HfApi, hf_hub_download

MANIFEST_FORMAT = "vxpu-manifest/v1alpha1"


def _fetch_range(url, start, length):
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


def build_manifest(repo_id, revision="main"):
    """Build a manifest for a Hugging Face model repo.

    Downloads no weight bytes: only file metadata and safetensors
    headers (a few KB per shard, via HTTP range requests).
    """
    info = HfApi().model_info(repo_id, revision=revision,
                              files_metadata=True)
    commit_sha = getattr(info, "sha", None) or revision
    with open(hf_hub_download(repo_id, "config.json",
                              revision=commit_sha)) as f:
        config = json.load(f)

    files, tensors = {}, {}
    shards = [s for s in info.siblings
              if s.rfilename.endswith(".safetensors")]
    for shard in shards:
        sha = shard.lfs.sha256
        source = (f"https://huggingface.co/{repo_id}/resolve/"
                  f"{commit_sha}/{shard.rfilename}")
        files[sha] = {"size": shard.size, "name": shard.rfilename,
                      "source": source}
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
