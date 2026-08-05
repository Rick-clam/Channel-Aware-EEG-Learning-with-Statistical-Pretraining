#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEEDS="${SEEDS:-2026 2027 2028}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESULT_DIR="${RESULT_DIR:-results/expert_screen}"
WIDE_DIM="${WIDE_DIM:?Set WIDE_DIM from the no-mixer expert-control audit}"

variants=(
  single_k7
  duplicate_k7
  multiscale_k3_7_15
  wide_single_k7
)
kernels=(
  7
  7,7,7
  3,7,15
  7
)
dims=(
  100
  100
  100
  "${WIDE_DIM}"
)

mkdir -p "${RESULT_DIR}" logs
for seed in ${SEEDS}; do
  for index in "${!variants[@]}"; do
    variant="${variants[$index]}"
    expert_kernels="${kernels[$index]}"
    feature_out="${dims[$index]}"
    run_id="sealedv3_TUEV_expert_${variant}_none_uniform_seed${seed}"
    result_path="${RESULT_DIR}/${run_id}.json"
    validation_path="${RESULT_DIR}/${run_id}_validation.json"

    if [[ -f "${result_path}" ]] && python - "${result_path}" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
valid = (
    record.get("status") == "completed"
    and record.get("config", {}).get("evaluation_mode") == "validation_only"
    and record.get("diagnostics", {}).get("test_set_accessed") is False
    and record.get("diagnostics", {}).get("test_loader_constructed") is False
)
raise SystemExit(0 if valid else 1)
PY
    then
      echo "SKIP completed validation-only result: ${run_id}"
    else
      EXPERT_AXIS=temporal \
      NORM_TYPE=local_filter \
      RELATION_MODE=none \
      GATE_DESCRIPTOR_SOURCE=raw \
      MATCH_DYNAMIC_BUDGET=0 \
      EXPERT_KERNELS="${expert_kernels}" \
      FEATURE_OUT="${feature_out}" \
      EVALUATION_MODE=validation_only \
      RESULT_DIR="${RESULT_DIR}" \
      EPOCHS="${EPOCHS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      NUM_WORKERS="${NUM_WORKERS}" \
      RUN_ID="${run_id}" \
      bash scripts/run_gate_variant.sh TUEV uniform "${seed}" "${GPU}"
    fi

    python tools/validate_validation_only_result.py \
      "${result_path}" --output "${validation_path}"
  done
done

python tools/summarize_tuev_expert_screen.py \
  --result_dir "${RESULT_DIR}" \
  --seeds ${SEEDS} \
  --wide_dim "${WIDE_DIM}" \
  --num_bootstrap 10000 \
  --bootstrap_seed 42026 \
  --output "${RESULT_DIR}/tuev_expert_screen_3seed_summary.json"
