#!/usr/bin/env bash
set -euo pipefail

cd /home/xr/refine_word
export PATH=/home/miniconda3/envs/biot/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

seed="${1:-2026}"
steps="${PRETRAIN_STEPS:-2000}"
transfer_mode="${TRANSFER_MODE:-frozen_linear}"
label_fraction="${LABEL_FRACTION:-1.0}"
batch_size="${TRANSFER_BATCH_SIZE:-128}"
workers="${TRANSFER_WORKERS:-8}"
epochs="${TRANSFER_EPOCHS:-30}"
pretrain_dir="results/pretrain_tuev_screen"
result_dir="results/pretrain_transfer_tuev"
mkdir -p "$result_dir" logs

variants=(scratch unmasked_stats normalized_masked_stats shuffled_masked_stats masked_waveform)
for variant in "${variants[@]}"; do
  pretrain_path=""
  if [[ "$variant" != "scratch" ]]; then
    pretrain_result="$pretrain_dir/tuev_holdout_${variant}_steps${steps}_seed${seed}.json"
    python -B tools/validate_pretrain_result.py \
      --result "$pretrain_result" --objective "$variant" \
      --seed "$seed" --max_steps "$steps" >/dev/null
    pretrain_path="$(python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["diagnostics"]["selected_checkpoint"]["path"])' "$pretrain_result")"
  fi
  fraction_tag="${label_fraction//./p}"
  run_id="tuev_${variant}_${transfer_mode}_labels${fraction_tag}_seed${seed}"
  result="$result_dir/$run_id.json"
  if [[ -s "$result" ]]; then
    python -B tools/validate_tuev_transfer_result.py \
      --result "$result" --variant "$variant" \
      --transfer_mode "$transfer_mode" --label_fraction "$label_fraction" \
      --seed "$seed"
    continue
  fi
  command=(
    python -B run_multi_moe.py
    --dataset TUEV
    --model dwmoespace_newgate
    --n_classes 6
    --feature_out 299
    --expert_axis temporal
    --expert_kernels 7
    --gate_type uniform
    --relation_mode none
    --norm_type local_filter
    --transfer_mode "$transfer_mode"
    --label_fraction "$label_fraction"
    --epochs "$epochs"
    --lr 0.001
    --batch_size "$batch_size"
    --num_workers "$workers"
    --checkpoint_metric val_cohen
    --evaluation_mode validation_only
    --deterministic
    --seed "$seed"
    --run_id "$run_id"
    --result_dir "$result_dir"
  )
  if [[ -n "$pretrain_path" ]]; then
    command+=(--pretrain_model_path "$pretrain_path")
  fi
  "${command[@]}" > "logs/$run_id.log" 2>&1
  python -B tools/validate_tuev_transfer_result.py \
    --result "$result" --variant "$variant" \
    --transfer_mode "$transfer_mode" --label_fraction "$label_fraction" \
    --seed "$seed"
done
