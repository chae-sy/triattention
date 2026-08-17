#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

DRY_RUN="${DRY_RUN:-0}"
JOB_PARALLEL="${JOB_PARALLEL:-1}"

GLOBAL_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
    GLOBAL_ARGS+=("--dry-run")
fi

MODEL="DeepSeek-R1-Distill-Qwen-7B"
INPUT="${RKV_ROOT}/triattention/calibration/aime24_calibration.txt"

python "${RKV_ROOT}/scripts/cli.py" "${GLOBAL_ARGS[@]}" \
    build-stats \
    --input "${INPUT}" \
    --model "${MODEL}" \
    --job-parallel "${JOB_PARALLEL}"