#!/usr/bin/env bash
set -euo pipefail

cd /home/xr/refine_word
export PATH=/home/miniconda3/envs/biot/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

seed="${1:-2026}"
steps="${PRETRAIN_STEPS:-2000}"
batch_size="${PRETRAIN_BATCH_SIZE:-128}"
workers="${PRETRAIN_WORKERS:-8}"
samples_per_dataset="${PRETRAIN_NORM_SAMPLES:-20000}"
objective="normalized_masked_stats"
result_dir="results/pretrain_stat_tuab_chbmit"
mkdir -p "$result_dir" results/manifests logs

run_one() {
  local target="$1"
  local logdataset="$2"
  local normalization="results/manifests/${logdataset}_stat_normalization_20k_v1.json"
  local run_id="${target}_holdout_${objective}_steps${steps}_seed${seed}"
  local result="${result_dir}/${run_id}.json"

  if [[ ! -s "$normalization" ]]; then
    python -B tools/fit_pretrain_stat_normalization.py \
      --logdataset "$logdataset" \
      --samples_per_dataset "$samples_per_dataset" \
      --batch_size "$batch_size" \
      --num_workers "$workers" \
      --output "$normalization" \
      > "logs/fit_${logdataset}_stat_normalization_20k_v1.log" 2>&1
  fi

  if [[ ! -s "$result" ]]; then
    python -B run_unsupervised_pretrain.py \
      --logdataset "$logdataset" \
      --objective "$objective" \
      --stat_normalization "$normalization" \
      --max_steps "$steps" \
      --epochs 100 \
      --checkpoint_every_n_steps 500 \
      --batch_size "$batch_size" \
      --num_workers "$workers" \
      --source_sampling balanced \
      --feature_out 100 \
      --gate_type static \
      --expert_axis legacy \
      --norm_type auto \
      --relation_mode dynamic_normalized \
      --expert_kernels 3,7,15 \
      --deterministic \
      --seed "$seed" \
      --run_id "$run_id" \
      --result_dir "$result_dir" \
      > "logs/${run_id}.log" 2>&1
  fi

  python -B tools/validate_pretrain_result.py \
    --result "$result" \
    --objective "$objective" \
    --seed "$seed" \
    --max_steps "$steps" \
    --logdataset "$logdataset" \
    --held_out_dataset "$target"
}

run_one tuab tuevchbmit
run_one chbmit tuabtuev
