#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset is required: TUAB, CHB_MIT, or TUEV}"
GATE="${2:?gate is required: uniform, static, or input}"
SEED="${3:?seed is required}"
GPU="${4:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESULT_DIR="${RESULT_DIR:-results/validation_screen}"
EXPERT_AXIS="${EXPERT_AXIS:-legacy}"
RELATION_MODE="${RELATION_MODE:-dynamic_normalized}"
NORM_TYPE="${NORM_TYPE:-auto}"
GATE_DESCRIPTOR_SOURCE="${GATE_DESCRIPTOR_SOURCE:-raw}"
MATCH_DYNAMIC_BUDGET="${MATCH_DYNAMIC_BUDGET:-0}"
EXPERT_KERNELS="${EXPERT_KERNELS:-3,7,15}"
FEATURE_OUT="${FEATURE_OUT:-100}"
EVALUATION_MODE="${EVALUATION_MODE:-validation_only}"
FINAL_CONFIG_LOCK="${FINAL_CONFIG_LOCK:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"

case "${DATASET}" in
  TUAB)
    SCRIPT="run_moe.py"
    N_CLASSES=1
    SAMPLE_LENGTH=10
    ;;
  CHB_MIT)
    SCRIPT="run_moe.py"
    N_CLASSES=1
    SAMPLE_LENGTH=10
    ;;
  TUEV)
    SCRIPT="run_multi_moe.py"
    N_CLASSES=6
    SAMPLE_LENGTH=5
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

KERNEL_TAG="${EXPERT_KERNELS//,/-}"
RUN_ID="${RUN_ID:-m1_${DATASET}_${EXPERT_AXIS}_${NORM_TYPE}_${RELATION_MODE}_${GATE}_k${KERNEL_TAG}_d${FEATURE_OUT}_seed${SEED}}"
mkdir -p "${RESULT_DIR}" logs

EXTRA_ARGS=()
if [[ "${MATCH_DYNAMIC_BUDGET}" == "1" ]]; then
  EXTRA_ARGS+=(--match_dynamic_budget)
fi
if [[ -n "${FINAL_CONFIG_LOCK}" ]]; then
  EXTRA_ARGS+=(--final_config_lock "${FINAL_CONFIG_LOCK}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
  --dataset "${DATASET}" \
  --model dwmoespace_newgate \
  --n_classes "${N_CLASSES}" \
  --sample_length "${SAMPLE_LENGTH}" \
  --gate_type "${GATE}" \
  --expert_axis "${EXPERT_AXIS}" \
  --norm_type "${NORM_TYPE}" \
  --relation_mode "${RELATION_MODE}" \
  --gate_descriptor_source "${GATE_DESCRIPTOR_SOURCE}" \
  --expert_kernels "${EXPERT_KERNELS}" \
  --feature_out "${FEATURE_OUT}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --run_id "${RUN_ID}" \
  --result_dir "${RESULT_DIR}" \
  --evaluation_mode "${EVALUATION_MODE}" \
  --limit_train_batches "${LIMIT_TRAIN_BATCHES}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --limit_test_batches "${LIMIT_TEST_BATCHES}" \
  --deterministic \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/${RUN_ID}.log"
