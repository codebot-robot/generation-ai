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

# torch JIT-compiles some eager CUDA ops via Triton at runtime,
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
# When the build context has no submodule checkout (CI checks out without
# submodules), the same commit is cloned instead. TRANSFORMERS_COMMIT must
# equal the submodule's gitlink; TestTransformersPinMatchesSubmodule checks.
ARG TRANSFORMERS_REPO=https://github.com/justinsb/transformers
ARG TRANSFORMERS_COMMIT=4096273a3a5b59ca169b2b81afc889ab62478ab8
COPY third_party/transformers/ /src/transformers/
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN if [ -f /src/transformers/pyproject.toml ]; then \
        TRANSFORMERS=/src/transformers; \
    else \
        TRANSFORMERS="git+${TRANSFORMERS_REPO}@${TRANSFORMERS_COMMIT}"; \
    fi \
    && pip install --no-cache-dir \
        torch==2.14.1 \
        "${TRANSFORMERS}" \
        grpcio \
        "protobuf>=6.30" \
        requests \
        huggingface_hub \
        accelerate

COPY python/vxpu/ /app/vxpu/

ENTRYPOINT ["python", "-m", "vxpu.server"]
