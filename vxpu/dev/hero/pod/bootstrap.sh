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

# Bootstrap a hero pod: packages, venv, copy from sibling pod, and doctor checks.
# Usage: bootstrap.sh [SIBLING_POD] | bootstrap.sh --check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSD="${VXPU_SSD_DIR:-/mnt/ssd}"
REQ="${REQUIREMENTS_FILE:-$SCRIPT_DIR/requirements.txt}"
SIBLING="${1:-${SIBLING_POD:-}}"
CHECK_ONLY=0; [ "$SIBLING" = "--check" ] && CHECK_ONLY=1 && SIBLING=
SUDO=; [ "$(id -u)" != 0 ] && SUDO=sudo

report_missing() {
  local missing=()
  [ -f "$SSD/env.sh" ] || missing+=("env: $SSD/env.sh")
  [ -x "$SSD/venv/bin/python" ] || missing+=("venv: $SSD/venv/bin/python")
  for d in src deepseek_v41 artifact blobs external; do
    [ -d "$SSD/$d" ] && [ -n "$(ls -A "$SSD/$d" 2>/dev/null)" ] || missing+=("$d: $SSD/$d")
  done
  for c in tmp hf xdg inductor triton aoti; do
    [ -d "$SSD/cache/$c" ] || missing+=("cache: $SSD/cache/$c")
  done

  if [ ${#missing[@]} -eq 0 ]; then
    echo "hero pod: all components present"
    return 0
  fi
  echo "hero pod: missing ${#missing[@]} component(s):"
  for m in "${missing[@]}"; do
    echo "  - $m"
  done
  return 1
}

[ "$CHECK_ONLY" -eq 1 ] && { report_missing || exit 1; exit 0; }

mkdir -p "$SSD"/{blobs,src,deepseek_v41,artifact,external,torch_ext} \
         "$SSD"/cache/{tmp,hf,xdg,inductor,triton,aoti}
[ -f "$SCRIPT_DIR/env.sh" ] && cp -p "$SCRIPT_DIR/env.sh" "$SSD/env.sh"

pkgs=()
for p in python3-venv python3-dev g++ liburing-dev git rsync curl; do
  dpkg -s "$p" >/dev/null 2>&1 || pkgs+=("$p")
done
if [ ${#pkgs[@]} -gt 0 ]; then
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${pkgs[@]}" >/dev/null
fi

[ -x "$SSD/venv/bin/python" ] || python3 -m venv "$SSD/venv"
if [ -f "$REQ" ] && [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  stamp="$SSD/venv/.req.sha256"
  hash=$(sha256sum "$REQ" | awk '{print $1}')
  if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$hash" ]; then
    "$SSD/venv/bin/pip" install -q --upgrade pip
    "$SSD/venv/bin/pip" install -q -r "$REQ"
    echo "$hash" > "$stamp"
  fi
fi

if [ -n "$SIBLING" ]; then
  for item in src deepseek_v41 artifact blobs cache; do
    if [ -d "$SIBLING" ]; then
      [ -d "$SIBLING/$item" ] && rsync -a "$SIBLING/$item/" "$SSD/$item/"
    else
      rsync -avz -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes" \
        "$SIBLING:$SSD/$item/" "$SSD/$item/" || echo "warning: could not sync $item from $SIBLING"
    fi
  done
fi

report_missing || true
