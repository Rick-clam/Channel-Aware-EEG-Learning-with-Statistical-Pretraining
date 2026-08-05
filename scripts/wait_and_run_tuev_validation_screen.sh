#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xr/refine_word}"
POLL_SECONDS="${POLL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
WAIT_LOG="${PROJECT_ROOT}/logs/tuev_relation_validation_screen_wait.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/validation_screen"
source /home/miniconda3/etc/profile.d/conda.sh
conda activate biot
cd "${PROJECT_ROOT}"
python tools/verify_sealed_evaluation_source.py \
  --output results/smoke/sealed_evaluation_source_verification.json
python tools/smoke_tuev_test_lock.py \
  --output results/smoke/tuev_test_lock_v2.json
python tools/smoke_binary_test_lock.py \
  --output results/smoke/binary_test_lock_v2.json
idle_checks=0
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  gpu_active="$(
    nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
  )"
  training_active="$(
    ps -eo pid=,args= \
      | grep -E '[p]ython(3)? .*([e]xperiments\.run_aggregation|[r]un_moe\.py|[r]un_multi_moe\.py|[r]un_unsupervised_pretrain\.py)' \
      | grep -v -- '--help' \
      | awk '{print $1}' | tr '\n' ',' || true
  )"
  queued_active="$(
    ps -eo pid=,args= \
      | grep -E '[b]ash .*[/]queue_.*after_.*\.sh' \
      | awk '{print $1}' | tr '\n' ',' || true
  )"
  if [[
    -z "${gpu_active}"
    && -z "${training_active}"
    && -z "${queued_active}"
  ]]; then
    idle_checks=$((idle_checks + 1))
    printf '%s ready/idle check %d/%d\n' \
      "$(date --iso-8601=seconds)" "${idle_checks}" "${REQUIRED_IDLE_CHECKS}" \
      >> "${WAIT_LOG}"
  else
    idle_checks=0
    printf '%s waiting; GPU=%s; training=%s; queued=%s\n' \
      "$(date --iso-8601=seconds)" \
      "${gpu_active:-none}" "${training_active:-none}" \
      "${queued_active:-none}" >> "${WAIT_LOG}"
  fi
  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    sleep "${POLL_SECONDS}"
  fi
done

if ! python tools/validate_validation_only_result.py \
  results/smoke/smoke4_TUEV_validation_only.json \
  --output results/smoke/smoke4_TUEV_validation_only_validation.json \
  >/dev/null 2>&1; then
EXPERT_AXIS=temporal \
NORM_TYPE=local_filter \
RELATION_MODE=dynamic_normalized \
GATE_DESCRIPTOR_SOURCE=raw \
EXPERT_KERNELS=3,7,15 \
FEATURE_OUT=100 \
EVALUATION_MODE=validation_only \
RESULT_DIR=results/smoke \
EPOCHS=1 \
BATCH_SIZE=16 \
NUM_WORKERS=2 \
LIMIT_TRAIN_BATCHES=0.005 \
LIMIT_VAL_BATCHES=0.01 \
LIMIT_TEST_BATCHES=0.01 \
RUN_ID=smoke4_TUEV_validation_only \
bash scripts/run_gate_variant.sh TUEV uniform 2026 0

fi

python tools/validate_validation_only_result.py \
  results/smoke/smoke4_TUEV_validation_only.json \
  --output results/smoke/smoke4_TUEV_validation_only_validation.json

SEEDS="2026 2027 2028" \
bash scripts/run_tuev_relation_validation_screen.sh
