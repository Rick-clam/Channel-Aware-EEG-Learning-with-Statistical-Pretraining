#!/usr/bin/env bash
set -euo pipefail

cd /home/xr/refine_word
export PATH=/home/miniconda3/envs/biot/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

seed="${1:-2026}"
steps="${PRETRAIN_STEPS:-2000}"
batch_size="${PRETRAIN_BATCH_SIZE:-128}"
workers="${PRETRAIN_WORKERS:-8}"
normalization="results/manifests/tuabchbmit_stat_normalization_20k_v1.json"
result_dir="results/pretrain_tuev_screen"
mkdir -p "$result_dir" logs

if [[ ! -s "$normalization" ]]; then
  echo "missing normalization artifact: $normalization" >&2
  exit 1
fi

objectives=(
  unmasked_stats
  normalized_masked_stats
  shuffled_masked_stats
  masked_waveform
)

for objective in "${objectives[@]}"; do
  run_id="tuev_holdout_${objective}_steps${steps}_seed${seed}"
  result="$result_dir/$run_id.json"
  if [[ -s "$result" ]]; then
    python -B tools/validate_pretrain_result.py \
      --result "$result" --objective "$objective" \
      --seed "$seed" --max_steps "$steps"
    continue
  fi
  command=(
    python -B run_unsupervised_pretrain.py
    --logdataset tuabchbmit
    --objective "$objective"
    --max_steps "$steps"
    --epochs 100
    --checkpoint_every_n_steps 500
    --batch_size "$batch_size"
    --num_workers "$workers"
    --source_sampling balanced
    --deterministic
    --seed "$seed"
    --run_id "$run_id"
    --result_dir "$result_dir"
  )
  if [[ "$objective" == "normalized_masked_stats" || "$objective" == "shuffled_masked_stats" ]]; then
    command+=(--stat_normalization "$normalization")
  fi
  "${command[@]}" > "logs/$run_id.log" 2>&1
  python -B tools/validate_pretrain_result.py \
    --result "$result" --objective "$objective" \
    --seed "$seed" --max_steps "$steps"
done
