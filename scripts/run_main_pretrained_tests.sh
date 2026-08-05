#!/usr/bin/env bash
set -euo pipefail

# Main leave-one-dataset-out pretrained downstream tests.
# Objective is fixed to normalized_masked_stats = mean + std + skewness only.
# Runs sequentially because LabServer3/4090-1 currently exposes one RTX 4090.

SEED="${1:-2026}"
GPU="${2:-0}"

cd /home/xr/refine_word
export PATH=/home/miniconda3/envs/biot/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

OBJECTIVE="normalized_masked_stats"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-2000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_WORKERS="${PRETRAIN_WORKERS:-8}"
PRETRAIN_NORM_SAMPLES="${PRETRAIN_NORM_SAMPLES:-20000}"

DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-100}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-128}"
DOWNSTREAM_WORKERS="${DOWNSTREAM_WORKERS:-8}"
TRANSFER_MODE="${TRANSFER_MODE:-full_finetune}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"

PRETRAIN_DENSE_DIR="results/pretrain_stat_tuab_chbmit"
PRETRAIN_TUEV_DIR="results/pretrain_tuev_screen"
VALIDATION_DIR="results/main_pretrained_downstream/validation"
TEST_DIR="results/main_pretrained_downstream/test"
LOCK_DIR="results/main_pretrained_downstream/locks"
SUMMARY_DIR="${LOCK_DIR}/summaries"
LOG_DIR="logs/main_pretrained_downstream"
mkdir -p \
  "${PRETRAIN_DENSE_DIR}" "${PRETRAIN_TUEV_DIR}" \
  "${VALIDATION_DIR}" "${TEST_DIR}" "${LOCK_DIR}" "${SUMMARY_DIR}" \
  "${LOG_DIR}" results/manifests logs

wait_for_gpu_idle() {
  if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
    return 0
  fi
  while ps -eo pid=,cmd= | grep -E "python -B run_(multi_)?moe\\.py" | grep -v grep >/dev/null; do
    date
    echo "waiting for existing run_moe/run_multi_moe process before starting the next main-pretrained run"
    sleep 300
  done
}

checkpoint_from_pretrain_result() {
  python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["diagnostics"]["selected_checkpoint"]["path"])' "$1"
}

fit_stat_normalization_if_missing() {
  local logdataset="$1"
  local normalization="results/manifests/${logdataset}_stat_normalization_20k_v1.json"
  if [[ ! -s "${normalization}" ]]; then
    wait_for_gpu_idle
    python -B tools/fit_pretrain_stat_normalization.py \
      --logdataset "${logdataset}" \
      --samples_per_dataset "${PRETRAIN_NORM_SAMPLES}" \
      --batch_size "${PRETRAIN_BATCH_SIZE}" \
      --num_workers "${PRETRAIN_WORKERS}" \
      --output "${normalization}" \
      > "logs/fit_${logdataset}_stat_normalization_20k_v1.log" 2>&1
  fi
  echo "${normalization}"
}

ensure_dense_pretrain() {
  local target="$1"
  local logdataset="$2"
  local result="${PRETRAIN_DENSE_DIR}/${target}_holdout_${OBJECTIVE}_steps${PRETRAIN_STEPS}_seed${SEED}.json"
  local normalization
  normalization="$(fit_stat_normalization_if_missing "${logdataset}")"
  if [[ ! -s "${result}" ]]; then
    wait_for_gpu_idle
    CUDA_VISIBLE_DEVICES="${GPU}" python -B run_unsupervised_pretrain.py \
      --logdataset "${logdataset}" \
      --objective "${OBJECTIVE}" \
      --stat_normalization "${normalization}" \
      --max_steps "${PRETRAIN_STEPS}" \
      --epochs 100 \
      --checkpoint_every_n_steps 500 \
      --batch_size "${PRETRAIN_BATCH_SIZE}" \
      --num_workers "${PRETRAIN_WORKERS}" \
      --source_sampling balanced \
      --feature_out 100 \
      --gate_type static \
      --expert_axis legacy \
      --norm_type auto \
      --relation_mode dynamic_normalized \
      --expert_kernels 3,7,15 \
      --deterministic \
      --seed "${SEED}" \
      --run_id "${target}_holdout_${OBJECTIVE}_steps${PRETRAIN_STEPS}_seed${SEED}" \
      --result_dir "${PRETRAIN_DENSE_DIR}" \
      > "logs/${target}_holdout_${OBJECTIVE}_steps${PRETRAIN_STEPS}_seed${SEED}.log" 2>&1
  fi
  python -B tools/validate_pretrain_result.py \
    --result "${result}" \
    --objective "${OBJECTIVE}" \
    --seed "${SEED}" \
    --max_steps "${PRETRAIN_STEPS}" \
    --logdataset "${logdataset}" \
    --held_out_dataset "${target}" \
    --backbone dense_original >/dev/null
  echo "${result}"
}

ensure_tuev_pretrain() {
  local result="${PRETRAIN_TUEV_DIR}/tuev_holdout_${OBJECTIVE}_steps${PRETRAIN_STEPS}_seed${SEED}.json"
  local normalization
  normalization="$(fit_stat_normalization_if_missing tuabchbmit)"
  if [[ ! -s "${result}" ]]; then
    wait_for_gpu_idle
    CUDA_VISIBLE_DEVICES="${GPU}" python -B run_unsupervised_pretrain.py \
      --logdataset tuabchbmit \
      --objective "${OBJECTIVE}" \
      --stat_normalization "${normalization}" \
      --max_steps "${PRETRAIN_STEPS}" \
      --epochs 100 \
      --checkpoint_every_n_steps 500 \
      --batch_size "${PRETRAIN_BATCH_SIZE}" \
      --num_workers "${PRETRAIN_WORKERS}" \
      --source_sampling balanced \
      --feature_out 299 \
      --gate_type uniform \
      --expert_axis temporal \
      --norm_type local_filter \
      --relation_mode none \
      --expert_kernels 7 \
      --deterministic \
      --seed "${SEED}" \
      --run_id "tuev_holdout_${OBJECTIVE}_steps${PRETRAIN_STEPS}_seed${SEED}" \
      --result_dir "${PRETRAIN_TUEV_DIR}" \
      > "logs/tuev_holdout_${OBJECTIVE}_steps${PRETRAIN_STEPS}_seed${SEED}.log" 2>&1
  fi
  python -B tools/validate_pretrain_result.py \
    --result "${result}" \
    --objective "${OBJECTIVE}" \
    --seed "${SEED}" \
    --max_steps "${PRETRAIN_STEPS}" \
    --logdataset tuabchbmit \
    --held_out_dataset tuev \
    --backbone tuev_original >/dev/null
  echo "${result}"
}

run_binary_pretrained_main() {
  local dataset="$1"
  local pretrain_result="$2"
  local fold_tag=""
  local n_classes=1
  local sample_length=10
  local split_manifest="results/manifests/tuab_effective_split.json"
  local extra_args=()
  if [[ "${dataset}" == "CHB_MIT" ]]; then
    fold_tag="_fold${CHBMIT_FOLD_INDEX:-0}"
    split_manifest="${CHBMIT_GROUPED_MANIFEST:-results/manifests/chbmit_patient_nested_5fold_v3.json}"
    extra_args+=(--chbmit_grouped_manifest "${split_manifest}")
    extra_args+=(--chbmit_fold_index "${CHBMIT_FOLD_INDEX:-0}")
  fi
  local pretrain_path
  pretrain_path="$(checkpoint_from_pretrain_result "${pretrain_result}")"
  local base="main_pretrained_${dataset}_seed${SEED}${fold_tag}"
  local validation_result="${VALIDATION_DIR}/${base}_validation_only.json"
  local test_result="${TEST_DIR}/${base}_final_test.json"
  local summary="${SUMMARY_DIR}/${base}_summary.json"
  local lock="${LOCK_DIR}/${base}_final_config.json"

  local common_args=(
    --dataset "${dataset}"
    --model dwmoespace_newgate
    --n_classes "${n_classes}"
    --sample_length "${sample_length}"
    --gate_type static
    --expert_axis legacy
    --norm_type auto
    --relation_mode dynamic_normalized
    --gate_descriptor_source raw
    --expert_kernels 3,7,15
    --feature_out 100
    --seed "${SEED}"
    --epochs "${DOWNSTREAM_EPOCHS}"
    --batch_size "${DOWNSTREAM_BATCH_SIZE}"
    --num_workers "${DOWNSTREAM_WORKERS}"
    --checkpoint_metric auto
    --transfer_mode "${TRANSFER_MODE}"
    --limit_train_batches "${LIMIT_TRAIN_BATCHES:-1.0}"
    --limit_val_batches "${LIMIT_VAL_BATCHES:-1.0}"
    --limit_test_batches "${LIMIT_TEST_BATCHES:-1.0}"
    --deterministic
    --pretrain_model_path "${pretrain_path}"
    "${extra_args[@]}"
  )

  if [[ ! -s "${validation_result}" ]]; then
    wait_for_gpu_idle
    CUDA_VISIBLE_DEVICES="${GPU}" python -B run_moe.py \
      "${common_args[@]}" \
      --evaluation_mode validation_only \
      --run_id "${base}_validation_only" \
      --result_dir "${VALIDATION_DIR}" \
      > "${LOG_DIR}/${base}_validation_only.log" 2>&1
  fi
  python -B tools/validate_binary_result.py "${validation_result}"

  if [[ ! -s "${summary}" ]]; then
    python -B tools/create_preregistered_run_summary.py \
      --validation_result "${validation_result}" \
      --output "${summary}"
  fi
  if [[ ! -s "${lock}" ]]; then
    python -B tools/create_final_binary_config_lock.py \
      --selected_validation_result "${validation_result}" \
      --compared_validation_result "${validation_result}" \
      --selection_rule "Fixed main pretrained condition; validation selects only checkpoint, epoch, and binary threshold." \
      --selection_summary "${summary}" \
      --split_manifest "${split_manifest}" \
      --output "${lock}" >/dev/null
  fi
  if [[ ! -s "${test_result}" ]]; then
    wait_for_gpu_idle
    CUDA_VISIBLE_DEVICES="${GPU}" python -B run_moe.py \
      "${common_args[@]}" \
      --evaluation_mode final_test \
      --final_config_lock "${lock}" \
      --run_id "${base}_final_test" \
      --result_dir "${TEST_DIR}" \
      > "${LOG_DIR}/${base}_final_test.log" 2>&1
  fi
  python -B tools/validate_binary_result.py "${test_result}"
}

run_tuev_pretrained_main() {
  local pretrain_result="$1"
  local pretrain_path
  pretrain_path="$(checkpoint_from_pretrain_result "${pretrain_result}")"
  local base="main_pretrained_TUEV_seed${SEED}"
  local validation_result="${VALIDATION_DIR}/${base}_validation_only.json"
  local test_result="${TEST_DIR}/${base}_final_test.json"
  local summary="${SUMMARY_DIR}/${base}_summary.json"
  local lock="${LOCK_DIR}/${base}_final_config.json"

  local common_args=(
    --dataset TUEV
    --model dwmoespace_newgate
    --n_classes 6
    --feature_out 299
    --expert_axis temporal
    --expert_kernels 7
    --gate_type uniform
    --relation_mode none
    --norm_type local_filter
    --transfer_mode "${TRANSFER_MODE}"
    --label_fraction 1.0
    --epochs "${DOWNSTREAM_EPOCHS}"
    --lr "${TUEV_LR:-0.001}"
    --batch_size "${DOWNSTREAM_BATCH_SIZE}"
    --num_workers "${DOWNSTREAM_WORKERS}"
    --checkpoint_metric val_cohen
    --deterministic
    --seed "${SEED}"
    --pretrain_model_path "${pretrain_path}"
    --limit_train_batches "${LIMIT_TRAIN_BATCHES:-1.0}"
    --limit_val_batches "${LIMIT_VAL_BATCHES:-1.0}"
    --limit_test_batches "${LIMIT_TEST_BATCHES:-1.0}"
  )

  if [[ ! -s "${validation_result}" ]]; then
    wait_for_gpu_idle
    CUDA_VISIBLE_DEVICES="${GPU}" python -B run_multi_moe.py \
      "${common_args[@]}" \
      --evaluation_mode validation_only \
      --run_id "${base}_validation_only" \
      --result_dir "${VALIDATION_DIR}" \
      > "${LOG_DIR}/${base}_validation_only.log" 2>&1
  fi
  python -B tools/validate_tuev_main_result.py \
    --result "${validation_result}" \
    --mode validation_only \
    --transfer_mode "${TRANSFER_MODE}" \
    --label_fraction 1.0 \
    --seed "${SEED}"

  if [[ ! -s "${summary}" ]]; then
    python -B tools/create_preregistered_run_summary.py \
      --validation_result "${validation_result}" \
      --output "${summary}"
  fi
  if [[ ! -s "${lock}" ]]; then
    python -B tools/create_final_tuev_config_lock.py \
      --selected_validation_result "${validation_result}" \
      --compared_validation_result "${validation_result}" \
      --selection_rule "Fixed main pretrained condition; validation selects only checkpoint and epoch." \
      --selection_summary "${summary}" \
      --tuev_manifest results/manifests/tuev_subject_split.json \
      --mapping_summary results/manifests/tuev_processed_file_mapping_summary.json \
      --output "${lock}" >/dev/null
  fi
  if [[ ! -s "${test_result}" ]]; then
    wait_for_gpu_idle
    CUDA_VISIBLE_DEVICES="${GPU}" python -B run_multi_moe.py \
      "${common_args[@]}" \
      --evaluation_mode final_test \
      --final_config_lock "${lock}" \
      --run_id "${base}_final_test" \
      --result_dir "${TEST_DIR}" \
      > "${LOG_DIR}/${base}_final_test.log" 2>&1
  fi
  python -B tools/validate_tuev_main_result.py \
    --result "${test_result}" \
    --mode final_test \
    --transfer_mode "${TRANSFER_MODE}" \
    --label_fraction 1.0 \
    --seed "${SEED}"
}

tuab_pretrain_result="$(ensure_dense_pretrain tuab tuevchbmit)"
tuev_pretrain_result="$(ensure_tuev_pretrain)"
chbmit_pretrain_result="$(ensure_dense_pretrain chbmit tuabtuev)"

run_binary_pretrained_main TUAB "${tuab_pretrain_result}"
run_tuev_pretrained_main "${tuev_pretrain_result}"
run_binary_pretrained_main CHB_MIT "${chbmit_pretrain_result}"

python -B - <<'PY'
import glob, json
for path in sorted(glob.glob("results/main_pretrained_downstream/test/*.json")):
    record = json.load(open(path))
    print(path, record.get("run_id"), record.get("metrics"))
PY
