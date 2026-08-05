#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset is required: TUAB or CHB_MIT}"
CONDITION="${2:?condition is required: scratch or pretrained}"
SEED="${3:-2026}"
GPU="${4:-0}"

cd /home/xr/refine_word
export PATH=/home/miniconda3/envs/biot/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

case "${DATASET}" in
  TUAB)
    N_CLASSES=1
    SAMPLE_LENGTH=10
    SPLIT_MANIFEST="results/manifests/tuab_effective_split.json"
    PRETRAIN_RESULT="${PRETRAIN_RESULT:-results/pretrain_stat_tuab_chbmit/tuab_holdout_normalized_masked_stats_steps2000_seed${SEED}.json}"
    ;;
  CHB_MIT)
    N_CLASSES=1
    SAMPLE_LENGTH=10
    CHBMIT_FOLD_INDEX="${CHBMIT_FOLD_INDEX:-0}"
    CHBMIT_GROUPED_MANIFEST="${CHBMIT_GROUPED_MANIFEST:-results/manifests/chbmit_patient_nested_5fold_v3.json}"
    SPLIT_MANIFEST="${CHBMIT_GROUPED_MANIFEST}"
    PRETRAIN_RESULT="${PRETRAIN_RESULT:-results/pretrain_stat_tuab_chbmit/chbmit_holdout_normalized_masked_stats_steps2000_seed${SEED}.json}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

case "${CONDITION}" in
  scratch)
    PRETRAIN_MODEL_PATH=""
    ;;
  pretrained)
    if [[ ! -s "${PRETRAIN_RESULT}" ]]; then
      echo "missing pretraining result: ${PRETRAIN_RESULT}" >&2
      exit 3
    fi
    PRETRAIN_MODEL_PATH="$(
      python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["diagnostics"]["selected_checkpoint"]["path"])' \
        "${PRETRAIN_RESULT}"
    )"
    ;;
  *)
    echo "Unsupported condition: ${CONDITION}" >&2
    exit 2
    ;;
esac

VALIDATION_DIR="results/pretrain_downstream/validation"
TEST_DIR="results/pretrain_downstream/test"
LOCK_DIR="results/pretrain_downstream/locks"
SUMMARY_DIR="${LOCK_DIR}/summaries"
mkdir -p "${VALIDATION_DIR}" "${TEST_DIR}" "${LOCK_DIR}" "${SUMMARY_DIR}" logs/pretrain_downstream

FOLD_TAG=""
EXTRA_ARGS=()
if [[ "${DATASET}" == "CHB_MIT" ]]; then
  FOLD_TAG="_fold${CHBMIT_FOLD_INDEX}"
  EXTRA_ARGS+=(--chbmit_grouped_manifest "${CHBMIT_GROUPED_MANIFEST}")
  EXTRA_ARGS+=(--chbmit_fold_index "${CHBMIT_FOLD_INDEX}")
fi

BASE="stat_transfer_${DATASET}_${CONDITION}_seed${SEED}${FOLD_TAG}"
VALIDATION_RESULT="${VALIDATION_DIR}/${BASE}_validation_only.json"
TEST_RESULT="${TEST_DIR}/${BASE}_final_test.json"
SUMMARY="${SUMMARY_DIR}/${BASE}_summary.json"
LOCK="${LOCK_DIR}/${BASE}_final_config.json"

if [[ -e "${TEST_RESULT}" ]]; then
  echo "Refusing to repeat locked test access: ${TEST_RESULT}" >&2
  exit 4
fi
if [[ -e "${LOCK}" ]]; then
  echo "Refusing to reuse an existing final-test lock: ${LOCK}" >&2
  exit 4
fi

COMMON_ARGS=(
  --dataset "${DATASET}"
  --model dwmoespace_newgate
  --n_classes "${N_CLASSES}"
  --sample_length "${SAMPLE_LENGTH}"
  --gate_type static
  --expert_axis legacy
  --norm_type auto
  --relation_mode dynamic_normalized
  --gate_descriptor_source raw
  --expert_kernels 3,7,15
  --feature_out 100
  --seed "${SEED}"
  --epochs "${EPOCHS:-100}"
  --batch_size "${BATCH_SIZE:-128}"
  --num_workers "${NUM_WORKERS:-8}"
  --checkpoint_metric auto
  --transfer_mode "${TRANSFER_MODE:-full_finetune}"
  --limit_train_batches "${LIMIT_TRAIN_BATCHES:-1.0}"
  --limit_val_batches "${LIMIT_VAL_BATCHES:-1.0}"
  --limit_test_batches "${LIMIT_TEST_BATCHES:-1.0}"
  --deterministic
  "${EXTRA_ARGS[@]}"
)
if [[ -n "${PRETRAIN_MODEL_PATH}" ]]; then
  COMMON_ARGS+=(--pretrain_model_path "${PRETRAIN_MODEL_PATH}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -B run_moe.py \
  "${COMMON_ARGS[@]}" \
  --evaluation_mode validation_only \
  --run_id "${BASE}_validation_only" \
  --result_dir "${VALIDATION_DIR}" \
  > "logs/pretrain_downstream/${BASE}_validation_only.log" 2>&1

python -B tools/validate_binary_result.py "${VALIDATION_RESULT}"
python -B tools/create_preregistered_run_summary.py \
  --validation_result "${VALIDATION_RESULT}" \
  --output "${SUMMARY}"

python -B tools/create_final_binary_config_lock.py \
  --selected_validation_result "${VALIDATION_RESULT}" \
  --compared_validation_result "${VALIDATION_RESULT}" \
  --selection_rule "Fixed scratch/pretrained condition; validation selects only checkpoint, epoch, and binary threshold." \
  --selection_summary "${SUMMARY}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --output "${LOCK}" >/dev/null

CUDA_VISIBLE_DEVICES="${GPU}" python -B run_moe.py \
  "${COMMON_ARGS[@]}" \
  --evaluation_mode final_test \
  --final_config_lock "${LOCK}" \
  --run_id "${BASE}_final_test" \
  --result_dir "${TEST_DIR}" \
  > "logs/pretrain_downstream/${BASE}_final_test.log" 2>&1

python -B tools/validate_binary_result.py "${TEST_RESULT}"
