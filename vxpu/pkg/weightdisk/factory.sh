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

# This is the startup script of the weights disk factory VM (weightdisk.go,
# #407). It formats the disk attached as vxpu-weights, then copies the
# checkpoint's files onto it from the weight bucket, as blobs/<sha256>. The
# VM's metadata names the bucket (vxpu-bucket) and lists the files
# (vxpu-files, one "sha256 size" per line), and holds the index
# (vxpu-index). After the copy, the script writes the index, unmounts
# the disk and powers the VM off.
#
# The script reports progress in guest attributes under vxpu/: status is one
# of starting, copying, done or "failed: <reason>", and copy_seconds is how
# long the copy took. The coordinator polls them.

set -uo pipefail

md=http://metadata.google.internal/computeMetadata/v1/instance
attr() { curl -s -X PUT --data "$2" -H "Metadata-Flavor: Google" "$md/guest-attributes/vxpu/$1" >/dev/null; }
meta() { curl -sf -H "Metadata-Flavor: Google" "$md/attributes/$1"; }
fail() { attr status "failed: $1"; shutdown -h now; exit 1; }

attr status starting
dev=/dev/disk/by-id/google-vxpu-weights
for _ in $(seq 120); do [ -e "$dev" ] && break; sleep 1; done
[ -e "$dev" ] || fail "no disk attached as vxpu-weights"
mkfs.ext4 -q -F -m 0 -L vxpu-weights "$dev" || fail "mkfs"
mkdir -p /mnt/weights && mount "$dev" /mnt/weights || fail "mount"
mkdir -p /mnt/weights/blobs

BUCKET=$(meta vxpu-bucket) || fail "no vxpu-bucket metadata"
meta vxpu-files > /tmp/files || fail "no vxpu-files metadata"

attr status copying
start=$(date +%s)
awk -v b="$BUCKET" '{ print b "/_blobs/" $1 }' /tmp/files |
  gcloud storage cp -I /mnt/weights/blobs/ > /tmp/cp.log 2>&1 || fail "gcloud storage cp: $(tail -c 300 /tmp/cp.log)"
seconds=$(( $(date +%s) - start ))

meta vxpu-index > /mnt/weights/weights.json || fail "no vxpu-index metadata"

sync
umount /mnt/weights || fail "umount"
attr copy_seconds "$seconds"
attr status done
shutdown -h now
