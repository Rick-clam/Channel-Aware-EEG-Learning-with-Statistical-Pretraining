#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xr/refine_word}"
cd "${PROJECT_ROOT}"
source /home/miniconda3/etc/profile.d/conda.sh
conda activate biot

EXPERT_AXIS=temporal \
NORM_TYPE=local_filter \
RELATION_MODE=dynamic_normalized \
GATE_DESCRIPTOR_SOURCE=raw \
MATCH_DYNAMIC_BUDGET=1 \
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
RUN_ID=smoke4_TUEV_validation_only_v2 \
bash scripts/run_gate_variant.sh TUEV uniform 2026 0

python tools/validate_validation_only_result.py \
  results/smoke/smoke4_TUEV_validation_only_v2.json \
  --output results/smoke/smoke4_TUEV_validation_only_v2_validation.json
