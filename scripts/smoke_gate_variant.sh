#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-TUEV}"
GATE="${2:-input}"
SEED="${3:-2026}"
GPU="${4:-0}"
EXPERT_AXIS="${EXPERT_AXIS:-temporal}"
NORM_TYPE="${NORM_TYPE:-local_filter}"
RELATION_MODE="${RELATION_MODE:-dynamic_normalized}"
GATE_DESCRIPTOR_SOURCE="${GATE_DESCRIPTOR_SOURCE:-raw}"
EXPERT_KERNELS="${EXPERT_KERNELS:-3,7,15}"
FEATURE_OUT="${FEATURE_OUT:-100}"

case "${DATASET}" in
  TUAB)
    SCRIPT="run_moe.py"
    N_CLASSES=1
    ;;
  TUEV)
    SCRIPT="run_multi_moe.py"
    N_CLASSES=6
    ;;
  *)
    echo "Smoke test currently supports TUAB or TUEV." >&2
    exit 2
    ;;
esac

KERNEL_TAG="${EXPERT_KERNELS//,/-}"
RUN_ID="${RUN_ID:-smoke_${DATASET}_${EXPERT_AXIS}_${NORM_TYPE}_${RELATION_MODE}_${GATE}_k${KERNEL_TAG}_d${FEATURE_OUT}_seed${SEED}}"
mkdir -p results/smoke logs

CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
  --dataset "${DATASET}" \
  --model dwmoespace_newgate \
  --n_classes "${N_CLASSES}" \
  --gate_type "${GATE}" \
  --expert_axis "${EXPERT_AXIS}" \
  --norm_type "${NORM_TYPE}" \
  --relation_mode "${RELATION_MODE}" \
  --gate_descriptor_source "${GATE_DESCRIPTOR_SOURCE}" \
  --expert_kernels "${EXPERT_KERNELS}" \
  --feature_out "${FEATURE_OUT}" \
  --seed "${SEED}" \
  --epochs 1 \
  --batch_size 16 \
  --num_workers 2 \
  --limit_train_batches 0.005 \
  --limit_val_batches 0.01 \
  --limit_test_batches 0.01 \
  --run_id "${RUN_ID}" \
  --result_dir results/smoke \
  --evaluation_mode smoke_test \
  --deterministic \
  2>&1 | tee "logs/${RUN_ID}.log"
