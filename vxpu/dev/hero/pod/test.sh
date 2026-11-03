#!/bin/bash
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

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$(cd "$DIR/../../../.." && pwd)/.build/bin:$PATH"

# 1. Manifest dry-run and schema validation
NS="${KUBE_NAMESPACE:-workspace-github-gke-labs-generation-ai-510}"
if command -v kubectl >/dev/null 2>&1; then
  kubectl apply --dry-run=client -n "$NS" -f "$DIR/pod.yaml" 2>/dev/null || kubectl apply --dry-run=client -f "$DIR/pod.yaml"
  kubectl apply --dry-run=server -n "$NS" -f "$DIR/pod.yaml" 2>/dev/null || true
fi
if command -v kubeconform >/dev/null 2>&1; then
  kubeconform -summary "$DIR/pod.yaml"
elif command -v go >/dev/null 2>&1; then
  GOTOOLCHAIN=auto go run github.com/yannh/kubeconform/cmd/kubeconform@latest -summary "$DIR/pod.yaml"
fi

# 2. env.sh variables check
(
  # shellcheck source=/dev/null
  source "$DIR/env.sh"
  [ "$VXPU_BLOB_DIR" = "/mnt/ssd/blobs" ]
  [ "$VXPU_EXTERNAL_DIR" = "/mnt/ssd/external" ]
  [ "$HF_HOME" = "/mnt/ssd/cache/hf" ]
  [ "$TMPDIR" = "/mnt/ssd/cache/tmp" ]
  [ "$HF_HUB_OFFLINE" = "1" ]
  [ "$VXPU_STEP_TIMEOUT" = "300" ]
)

# 3. GPU-free containerized bootstrap test (idempotency, sibling copy, report missing)
tar -C "$DIR" -cf - . | docker run -i --rm ubuntu:24.04 bash -c '
  set -euo pipefail
  mkdir -p /pod /tmp/test-ssd && tar -C /pod -xf -
  export VXPU_SSD_DIR=/tmp/test-ssd SKIP_PIP_INSTALL=1

  # Initial run: reports missing pieces
  out=$(/pod/bootstrap.sh)
  echo "$out" | grep -q "missing 5 component(s)"
  echo "$out" | grep -q "src: /tmp/test-ssd/src"

  # Idempotent run: succeeds cleanly
  /pod/bootstrap.sh

  # Sibling sync
  mkdir -p /tmp/sib/{src,deepseek_v41,artifact,blobs,cache}
  touch /tmp/sib/src/a.py /tmp/sib/deepseek_v41/b.py /tmp/sib/artifact/c.json /tmp/sib/blobs/d.bin
  mkdir -p /tmp/test-ssd/external && touch /tmp/test-ssd/external/bank.bin
  /pod/bootstrap.sh /tmp/sib

  # Doctor check passes when all components are populated
  /pod/bootstrap.sh --check
'
echo "All hero pod tests passed."
