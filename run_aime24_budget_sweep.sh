#!/usr/bin/env bash
set -euo pipefail

# Sequential TriAttention AIME 2024 sweep at fixed KV budgets.
# Inference/prompt/sampling defaults come from:
#   triattention/configs/shared/runner_defaults.yaml

TRI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${TRI_ROOT}/scripts/cli.py"

DATASET="${DATASET:-aime24}"
MODEL="${MODEL:-DeepSeek-R1-Distill-Qwen-7B}"
STATS_PATH="${STATS_PATH:-${TRI_ROOT}/triattention/calibration/for_aime24_experiment/ds_qwen7b.pt}"
RUN_TAG="${RUN_TAG:-rpc_sampling42}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"
START="${START:-}"
END="${END:-}"

usage() {
  printf '%s\n' \
    'Usage: run_aime24_budget_sweep.sh [--start INDEX] [--end INDEX]' \
    '' \
    'Runs every configured budget over dataset indices [start, end).' \
    'START and END environment variables are also supported.' \
    '' \
    'Example:' \
    '  bash benchmark/triattention/run_aime24_budget_sweep.sh --start 0 --end 1'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      [[ $# -ge 2 ]] || { echo "--start requires a value" >&2; exit 2; }
      START="$2"
      shift 2
      ;;
    --end)
      [[ $# -ge 2 ]] || { echo "--end requires a value" >&2; exit 2; }
      END="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Override with, for example: BUDGETS="512 1024" bash run_aime24_budget_sweep.sh
read -r -a BUDGET_LIST <<< "${BUDGETS:-1024 512}"

if [[ ! -f "${CLI}" ]]; then
  echo "TriAttention CLI not found: ${CLI}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${STATS_PATH}" ]]; then
  echo "TriAttention stats not found: ${STATS_PATH}" >&2
  echo "Set STATS_PATH to the intended calibration .pt file." >&2
  exit 1
fi
if [[ ${#BUDGET_LIST[@]} -eq 0 ]]; then
  echo "BUDGETS must contain at least one integer." >&2
  exit 1
fi
for budget in "${BUDGET_LIST[@]}"; do
  if ! [[ "${budget}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid budget: ${budget}" >&2
    exit 1
  fi
done
if [[ -n "${START}" ]] && ! [[ "${START}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --start: ${START}" >&2
  exit 1
fi
if [[ -n "${END}" ]] && ! [[ "${END}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --end: ${END}" >&2
  exit 1
fi
if [[ -n "${START}" && -n "${END}" ]] && (( END <= START )); then
  echo "--end must be greater than --start" >&2
  exit 1
fi

export PYTHONPATH="${TRI_ROOT}:${PYTHONPATH:-}"

GLOBAL_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  GLOBAL_ARGS+=(--dry-run)
fi

for budget in "${BUDGET_LIST[@]}"; do
  echo "============================================================"
  echo "[TriAttention sweep] dataset=${DATASET} model=${MODEL} budget=${budget}"
  echo "[TriAttention sweep] stats=${STATS_PATH} run_tag=${RUN_TAG}"
  echo "[TriAttention sweep] range=[${START:-0}, ${END:-all})"
  echo "============================================================"

  ARGS=(
    --dataset "${DATASET}"
    --model "${MODEL}"
    --method triattention
    --budget "${budget}"
    --stats-path "${STATS_PATH}"
    --run-tag "${RUN_TAG}"
  )
  if [[ -n "${EXTRA_CONFIG}" ]]; then
    ARGS+=(--extra-config "${EXTRA_CONFIG}")
  fi
  if [[ -n "${START}" ]]; then
    ARGS+=(--start "${START}")
  fi
  if [[ -n "${END}" ]]; then
    ARGS+=(--end "${END}")
  fi

  python "${CLI}" "${GLOBAL_ARGS[@]}" run-one "${ARGS[@]}"
done

echo "[TriAttention sweep] completed budgets: ${BUDGET_LIST[*]}"
