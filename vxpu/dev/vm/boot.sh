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

# Bring the VM up as an external executor without SSH (#357): where the
# project's firewall leaves no SSH path (security policy removes custom
# rules), the VM sets itself up from a startup script. This packages the
# working tree's vxpu code and a bootstrap script, uploads them to
# gs://BUCKET/_dev/$VM/ (the VM reads the bucket with its read-only
# storage scope), sets the VM's startup script to fetch and run the
# bootstrap, and resets the VM. On boot the bootstrap runs
# remote/setup-vm.sh, unpacks the code, and starts the executor server as
# a systemd service with the local weight tier (VXPU_BLOB_DIR,
# VXPU_BLOB_BUCKET), logging to the serial console; this script follows
# the console until the executor listens and prints its address for
# `vxpu up --external-executor`. Run again after a code change (a reset
# keeps the local SSDs, so the weights and the environment stay).
#
# The executor is reachable from the cluster on the VM's internal IP; its
# progress and logs are on the serial console:
#   gcloud compute instances get-serial-port-output $VM --zone ZONE

set -euo pipefail
. "$(dirname "$0")/common.sh"

BUCKET=${BUCKET:-vxpu-$PROJECT-us-central1}
PORT=${PORT:-50051}
KEEP_ALIVE=${KEEP_ALIVE:-86400}
EXECUTOR_ENV=${EXECUTOR_ENV:-}  # extra KEY=VALUE settings for the executor, space-separated
z=$(zone)

TORCH=$(sed -n 's/^ *torch==\([0-9.]*\).*/\1/p' "$VXPU_DIR/images/executor/Dockerfile" | head -1)
pins="'$TORCH' '$(pin TRANSFORMERS_REPO)' '$(pin TRANSFORMERS_COMMIT)' '$(pin DEEPSEEK_V41_REPO)' '$(pin DEEPSEEK_V41_COMMIT)'"
prefix="gs://$BUCKET/_dev/$VM"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
COPYFILE_DISABLE=1 tar --exclude=__pycache__ -czf "$work/vxpu-src.tgz" -C "$VXPU_DIR/python" vxpu
cp "$(dirname "$0")/remote/setup-vm.sh" "$work/setup-vm.sh"
cat > "$work/bootstrap.sh" <<BOOT
#!/bin/bash
# written by boot.sh: set up the VM and start the executor (logs on the serial console)
set -euo pipefail
exec > >(tee -a /var/log/vxpu-bootstrap.log /dev/ttyS0) 2>&1
echo "[vxpu-boot] starting"
gcloud storage cp $prefix/setup-vm.sh $prefix/vxpu-src.tgz /tmp/
OWNER=root bash /tmp/setup-vm.sh $pins
rm -rf /mnt/ssd/src/vxpu && mkdir -p /mnt/ssd/src && tar -xzf /tmp/vxpu-src.tgz -C /mnt/ssd/src
systemctl stop vxpu-executor 2>/dev/null || true
systemctl reset-failed vxpu-executor 2>/dev/null || true
systemd-run --unit=vxpu-executor --property=StandardOutput=journal+console --property=StandardError=journal+console \\
  --property=Restart=on-failure --property=WorkingDirectory=/mnt/ssd \\
  bash -c "source /mnt/ssd/env.sh && export VXPU_BLOB_DIR=/mnt/ssd/blobs VXPU_BLOB_BUCKET=gs://$BUCKET $EXECUTOR_ENV && exec python -m vxpu.server --port $PORT --keep-alive $KEEP_ALIVE"
echo "[vxpu-boot] executor service started"
BOOT
gcloud storage cp "$work/bootstrap.sh" "$work/setup-vm.sh" "$work/vxpu-src.tgz" "$prefix/" >/dev/null
cat > "$work/startup.sh" <<STARTUP
#!/bin/bash
gcloud storage cp $prefix/bootstrap.sh /tmp/vxpu-bootstrap.sh && bash /tmp/vxpu-bootstrap.sh
STARTUP
gcloud compute instances add-metadata "$VM" --zone "$z" --project "$PROJECT" --metadata-from-file startup-script="$work/startup.sh" >/dev/null
started=$(date -u +%Y-%m-%dT%H:%M:%S)
echo "resetting $VM to run the bootstrap (code and bootstrap in $prefix/)"
gcloud compute instances reset "$VM" --zone "$z" --project "$PROJECT" --quiet >/dev/null

# follow the serial console until the executor listens (or the bootstrap fails)
deadline=$(( $(date +%s) + ${BOOT_TIMEOUT:-2400} ))
seen=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 20
  out=$(gcloud compute instances get-serial-port-output "$VM" --zone "$z" --project "$PROJECT" 2>/dev/null \
        | awk -v s="$started" '$1 >= s' || true)
  lines=$(echo "$out" | grep -E "\[vxpu-boot\]|\[vxpu\]|python environment ok|packages ok|Traceback|Error" || true)
  n=$(echo "$lines" | grep -c . || true)
  if [ "$n" -gt "$seen" ]; then echo "$lines" | tail -n $(( n - seen )); seen=$n; fi
  if echo "$out" | grep -q "\[vxpu\] executor on"; then
    ip=$(gcloud compute instances describe "$VM" --zone "$z" --project "$PROJECT" --format='value(networkInterfaces[0].networkIP)')
    echo "executor on $VM at $ip:$PORT"
    echo "  vxpu up --external-executor $ip:$PORT"
    exit 0
  fi
done
echo "the executor did not come up in time; see the serial console" >&2
exit 1
