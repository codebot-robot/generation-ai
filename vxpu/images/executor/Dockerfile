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

FROM python:3.13-slim

WORKDIR /app

# torch 2.13 JIT-compiles some eager CUDA ops via Triton at runtime,
# which needs a C compiler even without torch.compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# torch is pinned to the exporting torch version: serialized
# ExportedPrograms promise newer-loads-older, but same-version is the
# tested configuration. The default Linux wheels include CUDA, so one
# image serves CPU and GPU nodes.
# transformers comes from the fork under third_party/ (a submodule), so the
# executor runs the same code the exporter traced; see docs/representation.md.
COPY third_party/transformers/ /src/transformers/
RUN pip install --no-cache-dir \
    torch==2.13.0 \
    /src/transformers \
    grpcio \
    "protobuf>=6.30" \
    requests \
    huggingface_hub

COPY python/vxpu/ /app/vxpu/

ENTRYPOINT ["python", "-m", "vxpu.server"]
