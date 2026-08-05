#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset is required: TUEV, TUAB, or CHB_MIT}"
TOP_K="${2:?Top-K is required: 1, 2, or 3}"
SEED="${3:?seed is required}"
GPU="${4:-0}"

if [[ ! "${TOP_K}" =~ ^[123]$ ]]; then
  echo "Top-K must be 1, 2, or 3" >&2
  exit 2
fi

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FEATURE_OUT="${FEATURE_OUT:-100}"
ROUTER_HIDDEN_DIM="${ROUTER_HIDDEN_DIM:-32}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.0}"
ROUTER_AUX_LOSS_COEF="${ROUTER_AUX_LOSS_COEF:-0.01}"
EXPERT_KERNELS="${EXPERT_KERNELS:-3,7,15}"
EVALUATION_MODE="${EVALUATION_MODE:-validation_only}"
RESULT_DIR="${RESULT_DIR:-results/sparse_topk/${EVALUATION_MODE}}"
FINAL_CONFIG_LOCK="${FINAL_CONFIG_LOCK:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"
CHBMIT_FOLD_INDEX="${CHBMIT_FOLD_INDEX:-0}"
CHBMIT_GROUPED_MANIFEST="${CHBMIT_GROUPED_MANIFEST:-results/manifests/chbmit_patient_nested_5fold_v3.json}"

case "${DATASET}" in
  TUEV)
    SCRIPT="run_multi_moe.py"
    N_CLASSES=6
    SAMPLE_LENGTH=5
    FOLD_TAG=""
    ;;
  TUAB)
    SCRIPT="run_moe.py"
    N_CLASSES=1
    SAMPLE_LENGTH=10
    FOLD_TAG=""
    ;;
  CHB_MIT)
    SCRIPT="run_moe.py"
    N_CLASSES=1
    SAMPLE_LENGTH=10
    FOLD_TAG="_fold${CHBMIT_FOLD_INDEX}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-sparse_topk_${DATASET}_k${TOP_K}_seed${SEED}${FOLD_TAG}_${EVALUATION_MODE}}"
LOG_DIR="logs/sparse_topk"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

EXTRA_ARGS=()
if [[ -n "${FINAL_CONFIG_LOCK}" ]]; then
  EXTRA_ARGS+=(--final_config_lock "${FINAL_CONFIG_LOCK}")
fi
if [[ "${DATASET}" == "CHB_MIT" ]]; then
  EXTRA_ARGS+=(
    --chbmit_grouped_manifest "${CHBMIT_GROUPED_MANIFEST}"
    --chbmit_fold_index "${CHBMIT_FOLD_INDEX}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU}" /home/miniconda3/envs/biot/bin/python "${SCRIPT}" \
  --dataset "${DATASET}" \
  --model dwmoespace_sparse_topk \
  --n_classes "${N_CLASSES}" \
  --sample_length "${SAMPLE_LENGTH}" \
  --feature_out "${FEATURE_OUT}" \
  --top_k "${TOP_K}" \
  --gate_type input \
  --gate_hidden_dim "${ROUTER_HIDDEN_DIM}" \
  --router_temperature "${ROUTER_TEMPERATURE}" \
  --router_aux_loss_coef "${ROUTER_AUX_LOSS_COEF}" \
  --expert_axis temporal \
  --norm_type local_filter \
  --relation_mode none \
  --gate_descriptor_source raw \
  --expert_kernels "${EXPERT_KERNELS}" \
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
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
