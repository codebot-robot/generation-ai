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

# Stage an artifact on the dev VM with its weights on local SSD.
#
#   fetch.sh ARTIFACT_DIR
#
# Copies every weight file the manifest references from the blob store's
# bucket (gs://$BUCKET/_blobs/<sha256>, the layout modelstore writes) to
# /mnt/ssd/blobs with gcloud's parallel client, then stages the artifact
# at /mnt/ssd/artifact with a manifest whose sources are those local
# files. Files already present are not fetched again.

set -euo pipefail
. "$(dirname "$0")/common.sh"

ART=${1:?usage: fetch.sh ARTIFACT_DIR}
BUCKET=${BUCKET:-vxpu-$PROJECT-us-central1}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

python3 - "$ART/manifest.json" "$BUCKET" "$work" <<'PY'
import json, sys
manifest, bucket, work = sys.argv[1:]
m = json.load(open(manifest))
with open(f"{work}/urls.txt", "w") as f:
    for sha, entry in m["files"].items():
        f.write(f"gs://{bucket}/_blobs/{sha} {entry['size']}\n")
for sha, entry in m["files"].items():
    entry["source"] = f"file:///mnt/ssd/blobs/{sha}"
json.dump(m, open(f"{work}/manifest.json", "w"))
print(f"{len(m['files'])} files, {sum(e['size'] for e in m['files'].values()) / 1e9:.1f} GB")
PY

vm_scp "$work/urls.txt" /tmp/vxpu-urls.txt
# a blob is present when its file has the manifest's size; the copy's
# failure, or any blob still missing after it, stops the script before
# the artifact is staged
vm_ssh 'set -e
missing() { while read -r url size; do f="/mnt/ssd/blobs/${url##*/}"; [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$size" ] || echo "$url"; done < /tmp/vxpu-urls.txt; }
todo=$(missing)
if [ -n "$todo" ]; then
  start=$(date +%s)
  if ! echo "$todo" | gcloud storage cp -I /mnt/ssd/blobs/ > /tmp/vxpu-cp.log 2>&1; then
    grep -v "^Copying" /tmp/vxpu-cp.log | tail -20 >&2
    echo "gcloud storage cp failed (log: /tmp/vxpu-cp.log on the VM)" >&2
    exit 1
  fi
  echo "fetched in $(( $(date +%s) - start )) s"
fi
still=$(missing)
if [ -n "$still" ]; then echo "$(echo "$still" | wc -l) blobs missing or short after the copy, e.g. $(echo "$still" | head -1)" >&2; exit 1; fi
echo "on local SSD: $(du -sh /mnt/ssd/blobs | cut -f1) in $(ls /mnt/ssd/blobs | wc -l) files, all present"'

for f in binding.json prefill.pt2 decode.pt2; do vm_scp "$ART/$f" "/mnt/ssd/artifact/$f"; done
vm_scp "$work/manifest.json" /mnt/ssd/artifact/manifest.json
echo "artifact staged at /mnt/ssd/artifact"
