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

# Prepare the dev VM: RAID0 its local SSDs at /mnt/ssd, install the
# packages the engine and its kernels need, and create a Python
# environment at the executor image's pins. Writes /mnt/ssd/env.sh.
# Re-running skips what is already done.

set -euo pipefail
. "$(dirname "$0")/common.sh"

TORCH=$(sed -n 's/^ *torch==\([0-9.]*\).*/\1/p' "$VXPU_DIR/images/executor/Dockerfile" | head -1)
TRANSFORMERS_REPO=$(pin TRANSFORMERS_REPO)
TRANSFORMERS_COMMIT=$(pin TRANSFORMERS_COMMIT)
DEEPSEEK_REPO=$(pin DEEPSEEK_V41_REPO)
DEEPSEEK_COMMIT=$(pin DEEPSEEK_V41_COMMIT)
echo "torch $TORCH, transformers $TRANSFORMERS_COMMIT, deepseek_v41 $DEEPSEEK_COMMIT"

# the steps run on the VM (remote/setup-vm.sh, also boot.sh's startup script)
vm_scp "$(dirname "$0")/remote/setup-vm.sh" /tmp/vxpu-setup-vm.sh
vm_ssh "bash /tmp/vxpu-setup-vm.sh '$TORCH' '$TRANSFORMERS_REPO' '$TRANSFORMERS_COMMIT' '$DEEPSEEK_REPO' '$DEEPSEEK_COMMIT'"
