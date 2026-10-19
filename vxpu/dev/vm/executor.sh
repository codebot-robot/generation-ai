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

# Run the executor server on the dev VM as an external executor for the
# router (#357): models registered by name are placed on it, and its
# local SSD is its weight tier (VXPU_BLOB_DIR=/mnt/ssd/blobs, filled from
# the weight bucket as models need files). Restarts the server if it is
# already running. Prints the address for `vxpu up --external-executor`.
#
#   executor.sh            start (or restart) the executor
#   executor.sh stop       stop it
#   executor.sh log        follow its log
#
# Run setup.sh and sync.sh first. The router reaches the VM on its
# internal IP (GKE pods and VMs share the project's VPC).

set -euo pipefail
. "$(dirname "$0")/common.sh"

PORT=${PORT:-50051}
BUCKET=${BUCKET:-vxpu-$PROJECT-us-central1}
KEEP_ALIVE=${KEEP_ALIVE:-86400}

stop='if [ -f /mnt/ssd/executor.pid ]; then kill "$(cat /mnt/ssd/executor.pid)" 2>/dev/null || true; rm -f /mnt/ssd/executor.pid; sleep 2; fi'
case "${1:-start}" in
  stop)
    vm_ssh "$stop; echo stopped"
    exit 0 ;;
  log)
    vm_ssh "tail -f /mnt/ssd/executor.log"
    exit 0 ;;
  start) ;;
  *) echo "usage: executor.sh [start|stop|log]" >&2; exit 2 ;;
esac

vm_ssh "set -e; $stop
source /mnt/ssd/env.sh
export VXPU_BLOB_DIR=/mnt/ssd/blobs VXPU_BLOB_BUCKET=gs://$BUCKET
cd /mnt/ssd
nohup python -m vxpu.server --port $PORT --keep-alive $KEEP_ALIVE > /mnt/ssd/executor.log 2>&1 &
echo \$! > /mnt/ssd/executor.pid
sleep 20
if ! kill -0 \"\$(cat /mnt/ssd/executor.pid)\" 2>/dev/null; then tail -20 /mnt/ssd/executor.log >&2; exit 1; fi
grep '^\[vxpu\] executor on' /mnt/ssd/executor.log || true"
ip=$(gcloud compute instances describe "$VM" --zone "$(zone)" --project "$PROJECT" --format='value(networkInterfaces[0].networkIP)')
echo "executor on $VM at $ip:$PORT"
echo "  vxpu up --external-executor $ip:$PORT"
