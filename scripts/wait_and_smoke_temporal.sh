#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xr/refine_word}"
POLL_SECONDS="${POLL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-smoke_TUEV_temporal_local_filter_dynamic_normalized_uniform_k3-7-15_d100_seed2026}"
WAIT_LOG="${PROJECT_ROOT}/logs/${RUN_ID}_wait.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/smoke"
idle_checks=0

while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  gpu_active="$(
    nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]'
  )"
  training_active="$(
    ps -eo pid=,args= \
      | grep -E \
        '[p]ython(3)? .*([e]xperiments\.run_aggregation|[r]un_moe\.py|[r]un_multi_moe\.py|[r]un_unsupervised_pretrain\.py)' \
      | grep -v -- '--help' \
      | awk '{print $1}' \
      | tr '\n' ',' \
      || true
  )"
  queued_active="$(
    ps -eo pid=,args= \
      | grep -E '[b]ash .*[/]queue_.*after_.*\.sh' \
      | awk '{print $1}' \
      | tr '\n' ',' \
      || true
  )"
  if [[
    -z "${gpu_active}"
    && -z "${training_active}"
    && -z "${queued_active}"
  ]]; then
    idle_checks=$((idle_checks + 1))
    printf '%s GPU idle check %d/%d\n' \
      "$(date --iso-8601=seconds)" "${idle_checks}" "${REQUIRED_IDLE_CHECKS}" \
      >> "${WAIT_LOG}"
  else
    idle_checks=0
    printf '%s waiting; GPU PIDs: %s; training PIDs: %s; queued PIDs: %s\n' \
      "$(date --iso-8601=seconds)" \
      "${gpu_active:-none}" "${training_active:-none}" \
      "${queued_active:-none}" >> "${WAIT_LOG}"
  fi
  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    sleep "${POLL_SECONDS}"
  fi
done

cd "${PROJECT_ROOT}"
source /home/miniconda3/etc/profile.d/conda.sh
conda activate biot

EXPERT_AXIS=temporal \
NORM_TYPE=local_filter \
RELATION_MODE=dynamic_normalized \
GATE_DESCRIPTOR_SOURCE=raw \
EXPERT_KERNELS=3,7,15 \
FEATURE_OUT=100 \
RUN_ID="${RUN_ID}" \
bash scripts/smoke_gate_variant.sh TUEV uniform 2026 "${GPU}"
