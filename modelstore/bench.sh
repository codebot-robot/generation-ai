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

set -o errexit
set -o nounset
set -o pipefail

# Benchmark harness for blob read throughput.
# Measures reading 16 GB in 64 MB ranges with 16 and 64 in flight
# against modelstore or against GCS directly.
#
# Usage:
#   TOKEN="<token>" ./bench.sh <URL>
#
# Examples:
#   ./bench.sh http://modelstore.modelstore/blobs/83bb444edc7d4a1ed91ab67ad19620e2ed74724cdd62ede330fcaa7748c64404
#   TOKEN=$(gcloud auth print-access-token) ./bench.sh https://storage.googleapis.com/my-bucket/_blobs/83bb444edc7d4a1ed91ab67ad19620e2ed74724cdd62ede330fcaa7748c64404

if [[ $# -ne 1 ]]; then
  echo "Usage: [TOKEN=...] $0 <URL>" >&2
  exit 1
fi

URL="$1"
TOKEN="${TOKEN:-}"

TOTAL_GB="${TOTAL_GB:-16}"
RANGE_MB="${RANGE_MB:-64}"
RANGE_SIZE=$((RANGE_MB * 1024 * 1024))
TOTAL_RANGES=$((TOTAL_GB * 1024 / RANGE_MB))

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

CURL_CONFIG="${TMP_DIR}/curl.cfg"
touch "${CURL_CONFIG}"
chmod 0600 "${CURL_CONFIG}"

if [[ -n "${TOKEN}" ]]; then
  echo "header = \"Authorization: Bearer ${TOKEN}\"" > "${CURL_CONFIG}"
fi

CURL_ARGS=("-s" "-f" "-o" "/dev/null" "-K" "${CURL_CONFIG}")

echo "Probing ${URL}..."
HEAD_OUTPUT=$(curl -s -f -I -K "${CURL_CONFIG}" "${URL}")
CONTENT_LENGTH=$(echo "${HEAD_OUTPUT}" | awk 'tolower($1) == "content-length:" {gsub("\r", ""); print $2}' | tail -n 1)

OBJECT_WINDOW=$((4 * 1024 * 1024 * 1024))
if [[ -n "${CONTENT_LENGTH}" && "${CONTENT_LENGTH}" -gt 0 ]]; then
  if [[ "${CONTENT_LENGTH}" -lt "${OBJECT_WINDOW}" ]]; then
    OBJECT_WINDOW="${CONTENT_LENGTH}"
  fi
  echo "Object size: ${CONTENT_LENGTH} bytes (benchmark window: ${OBJECT_WINDOW} bytes)"
else
  echo "Warning: could not determine Content-Length; defaulting benchmark window to ${OBJECT_WINDOW} bytes"
fi

if [[ "${RANGE_SIZE}" -gt "${OBJECT_WINDOW}" ]]; then
  RANGE_SIZE="${OBJECT_WINDOW}"
  RANGE_MB=$((RANGE_SIZE / 1024 / 1024))
  if [[ "${RANGE_MB}" -eq 0 ]]; then
    RANGE_MB=1
  fi
  TOTAL_RANGES=$((TOTAL_GB * 1024 / RANGE_MB))
  echo "Adjusted range size to ${RANGE_MB} MB (${RANGE_SIZE} bytes) to fit object window"
fi

echo "Benchmarking ${URL}"
echo "Planned transfer: ~${TOTAL_GB} GB in ${TOTAL_RANGES} ranges of ${RANGE_MB} MB each"

# Generate range specifications
RANGES_FILE="${TMP_DIR}/ranges.txt"
WINDOW_RANGES=$((OBJECT_WINDOW / RANGE_SIZE))
if [[ "${WINDOW_RANGES}" -le 0 ]]; then
  WINDOW_RANGES=1
fi

for ((i = 0; i < TOTAL_RANGES; i++)); do
  SLOT=$((i % WINDOW_RANGES))
  START=$((SLOT * RANGE_SIZE))
  END=$((START + RANGE_SIZE - 1))
  echo "${START}-${END}" >> "${RANGES_FILE}"
done

for CONCURRENCY in 16 64; do
  SIZES_FILE="${TMP_DIR}/sizes_${CONCURRENCY}.txt"
  START_TIME=$(date +%s%N)

  # Run concurrent range requests with curl, capturing downloaded size and http status code per request.
  XARGS_STATUS=0
  xargs -P "${CONCURRENCY}" -a "${RANGES_FILE}" -I {} \
    curl "${CURL_ARGS[@]}" -w '%{size_download} %{http_code}\n' -r "{}" "${URL}" > "${SIZES_FILE}" || XARGS_STATUS=$?

  END_TIME=$(date +%s%N)

  NANOS=$((END_TIME - START_TIME))
  DURATION_SEC=$(awk "BEGIN {print ${NANOS} / 1000000000}")

  TOTAL_BYTES=$(awk '{s += $1} END {printf "%.0f", s}' "${SIZES_FILE}")
  SUCCESS_COUNT=$(awk '$2 ~ /^2/ {c++} END {print c+0}' "${SIZES_FILE}")
  FAILED_COUNT=$((TOTAL_RANGES - SUCCESS_COUNT))

  ACTUAL_GB=$(awk "BEGIN {printf \"%.2f\", ${TOTAL_BYTES} / (1024 * 1024 * 1024)}")
  THROUGHPUT=$(awk "BEGIN {if (${DURATION_SEC} > 0) printf \"%.2f\", ${ACTUAL_GB} / ${DURATION_SEC}; else print \"0.00\"}")

  if [[ "${FAILED_COUNT}" -gt 0 || "${XARGS_STATUS}" -ne 0 ]]; then
    echo "Concurrency ${CONCURRENCY} FAILED: ${FAILED_COUNT} of ${TOTAL_RANGES} requests failed (${ACTUAL_GB} GB in ${DURATION_SEC}s)" >&2
    exit 1
  fi

  echo "Concurrency ${CONCURRENCY}: ${ACTUAL_GB} GB in ${DURATION_SEC}s -> ${THROUGHPUT} GB/s"
done
