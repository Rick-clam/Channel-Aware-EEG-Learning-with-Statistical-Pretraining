#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset is required}"
TOP_K="${2:?Top-K is required}"
SEED="${3:?seed is required}"
GPU="${4:-0}"
CHBMIT_FOLD_INDEX="${CHBMIT_FOLD_INDEX:-0}"
CHBMIT_GROUPED_MANIFEST="${CHBMIT_GROUPED_MANIFEST:-results/manifests/chbmit_patient_nested_5fold_v3.json}"

case "${DATASET}" in
  TUEV|TUAB)
    FOLD_TAG=""
    ;;
  CHB_MIT)
    FOLD_TAG="_fold${CHBMIT_FOLD_INDEX}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

BASE="sparse_topk_${DATASET}_k${TOP_K}_seed${SEED}${FOLD_TAG}"
VALIDATION_DIR="results/sparse_topk/validation"
TEST_DIR="results/sparse_topk/test"
LOCK_DIR="results/sparse_topk/locks"
SUMMARY_DIR="${LOCK_DIR}/summaries"
VALIDATION_RESULT="${VALIDATION_DIR}/${BASE}_validation_only.json"
TEST_RESULT="${TEST_DIR}/${BASE}_final_test.json"
SUMMARY="${SUMMARY_DIR}/${BASE}_summary.json"
LOCK="${LOCK_DIR}/${BASE}_final_config.json"
mkdir -p "${VALIDATION_DIR}" "${TEST_DIR}" "${SUMMARY_DIR}"

if [[ -e "${TEST_RESULT}" ]]; then
  echo "Refusing to repeat locked test access: ${TEST_RESULT}" >&2
  exit 3
fi
if [[ -e "${LOCK}" ]]; then
  echo "Refusing to reuse an existing final-test lock: ${LOCK}" >&2
  exit 3
fi

export EVALUATION_MODE=validation_only
export RESULT_DIR="${VALIDATION_DIR}"
export RUN_ID="${BASE}_validation_only"
export CHBMIT_FOLD_INDEX CHBMIT_GROUPED_MANIFEST
scripts/run_sparse_topk_variant.sh "${DATASET}" "${TOP_K}" "${SEED}" "${GPU}"
/home/miniconda3/envs/biot/bin/python -B tools/validate_sparse_topk_result.py \
  "${VALIDATION_RESULT}"

/home/miniconda3/envs/biot/bin/python -B tools/create_preregistered_run_summary.py \
  --validation_result "${VALIDATION_RESULT}" \
  --output "${SUMMARY}"

SELECTION_RULE="Preregistered Top-K configuration; validation selects only checkpoint, epoch, and any binary threshold."
if [[ "${DATASET}" == "TUEV" ]]; then
  /home/miniconda3/envs/biot/bin/python -B tools/create_final_tuev_config_lock.py \
    --selected_validation_result "${VALIDATION_RESULT}" \
    --compared_validation_result "${VALIDATION_RESULT}" \
    --selection_rule "${SELECTION_RULE}" \
    --selection_summary "${SUMMARY}" \
    --output "${LOCK}" >/dev/null
else
  if [[ "${DATASET}" == "CHB_MIT" ]]; then
    SPLIT_MANIFEST="${CHBMIT_GROUPED_MANIFEST}"
  else
    SPLIT_MANIFEST="results/manifests/tuab_effective_split.json"
  fi
  /home/miniconda3/envs/biot/bin/python -B tools/create_final_binary_config_lock.py \
    --selected_validation_result "${VALIDATION_RESULT}" \
    --compared_validation_result "${VALIDATION_RESULT}" \
    --selection_rule "${SELECTION_RULE}" \
    --selection_summary "${SUMMARY}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --output "${LOCK}" >/dev/null
fi

export EVALUATION_MODE=final_test
export RESULT_DIR="${TEST_DIR}"
export RUN_ID="${BASE}_final_test"
export FINAL_CONFIG_LOCK="${LOCK}"
export LIMIT_TRAIN_BATCHES=1.0
export LIMIT_VAL_BATCHES=1.0
export LIMIT_TEST_BATCHES=1.0
scripts/run_sparse_topk_variant.sh "${DATASET}" "${TOP_K}" "${SEED}" "${GPU}"
/home/miniconda3/envs/biot/bin/python -B tools/validate_sparse_topk_result.py \
  "${TEST_RESULT}"
