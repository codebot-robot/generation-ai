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


"""Compile a model for this engine and store the result in modelstore (#357).

The engine's "query compiler": given a model name, export its artifact
(export.py: the weightless graphs, the manifest of content-addressed
weight references and the binding) and push it to modelstore as the
repository vxpu-artifacts/<model>, one layer per file, under the tag the
caller names. The router runs this in a Kubernetes Job with the executor
image, so the artifact is exported by exactly the torch and the forks
(transformers, deepseek_v41) the executors that run it have, on a CPU
node: export runs on torch's meta device and reads only the model's
config, metadata and safetensors headers, never the weights.

The model is read from modelstore through its Hub-compatible API
(HF_ENDPOINT); the router imports the weights there first. The tag is
the router's catalog entry: it names the model revision and the
executor image, and the manifest's annotations record the source model,
revision and commit (when the endpoint names one), and the compiler's
versions.

    python -m vxpu.compile REPO --modelstore URL --tag TAG [--revision main] [--max-cache-len 1024]

The last line printed is a JSON summary (reference, digest, commit).
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile


def compiler_versions():
    """What produced the artifact: torch and transformers versions, and a
    digest of the vxpu package's own sources (the exporter, the ops)."""
    import torch
    import transformers
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(here)):
        for name in sorted(files):
            if name.endswith(".py"):
                with open(os.path.join(root, name), "rb") as f:
                    h.update(name.encode() + b"\0" + f.read())
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "vxpu_sources": h.hexdigest()[:16]}
    fork = os.environ.get("VXPU_DEEPSEEK_V41")
    if fork and os.path.exists(os.path.join(fork, "model.py")):
        with open(os.path.join(fork, "model.py"), "rb") as f:
            versions["deepseek_v41_model"] = hashlib.sha256(f.read()).hexdigest()[:16]
    return versions


def main(argv=None):
    p = argparse.ArgumentParser(description="Compile a model for this engine and store it in modelstore.")
    p.add_argument("repo_id")
    p.add_argument("--modelstore", required=True, help="modelstore base URL (its Hub API is read, its registry written)")
    p.add_argument("--tag", required=True, help="tag for the compiled artifact (the router's catalog entry)")
    p.add_argument("--revision", default="main")
    p.add_argument("--max-cache-len", type=int, default=1024)
    args = p.parse_args(argv)

    # read the model through modelstore; huggingface_hub reads the
    # endpoint when it is first imported, so this precedes the imports
    os.environ["HF_ENDPOINT"] = args.modelstore.rstrip("/")
    from . import modelstore
    from .export import export_artifact

    name = modelstore.artifact_repository(args.repo_id)
    if not modelstore.TAG_RE.fullmatch(args.tag):
        raise SystemExit(f"--tag {args.tag!r} is not a valid tag")
    with tempfile.TemporaryDirectory(prefix="vxpu-compile-") as out:
        manifest, binding = export_artifact(args.repo_id, out, args.max_cache_len, args.revision)
        source = manifest["source"]
        annotations = {
            modelstore.ANNOTATION_SOURCE_REPO: args.repo_id,
            modelstore.ANNOTATION_SOURCE_REVISION: args.revision,
            "vnd.gke-labs.vxpu.compiler": json.dumps(compiler_versions(), sort_keys=True),
            "vnd.gke-labs.vxpu.max-cache-len": str(args.max_cache_len),
        }
        if source.get("commit_pinned"):  # a commit, not a moving ref
            annotations[modelstore.ANNOTATION_SOURCE_COMMIT] = source["commit_sha"]
        digest = modelstore.push_directory(args.modelstore, name, args.tag, out, annotations)
    summary = {"reference": f"{name}:{args.tag}", "digest": digest, "commit": source.get("commit_sha"),
               "weight_bytes": sum(f["size"] for f in manifest["files"].values()),
               "tensors": len(manifest["tensors"])}
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
