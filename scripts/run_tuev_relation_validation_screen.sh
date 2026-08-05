#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEEDS="${SEEDS:-2026 2027 2028}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESULT_DIR="${RESULT_DIR:-results/validation_screen}"

relations=(
  none
  attention
  static
  spatial1x1
  static_conditioned_matched
  spatial1x1_conditioned_matched
  dynamic
  dynamic_normalized
)

mkdir -p "${RESULT_DIR}" logs
for seed in ${SEEDS}; do
for relation in "${relations[@]}"; do
  run_id="sealedv2_TUEV_temporal_local_filter_${relation}_uniform_k3-7-15_d100_seed${seed}"
  result_path="${RESULT_DIR}/${run_id}.json"
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
    continue
  fi

  EXPERT_AXIS=temporal \
  NORM_TYPE=local_filter \
  RELATION_MODE="${relation}" \
  GATE_DESCRIPTOR_SOURCE=raw \
  MATCH_DYNAMIC_BUDGET=1 \
  EXPERT_KERNELS=3,7,15 \
  FEATURE_OUT=100 \
  EVALUATION_MODE=validation_only \
  RESULT_DIR="${RESULT_DIR}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  RUN_ID="${run_id}" \
  bash scripts/run_gate_variant.sh TUEV uniform "${seed}" "${GPU}"
done
done

python tools/summarize_tuev_relation_screen.py \
  --result_dir "${RESULT_DIR}" \
  --seeds ${SEEDS} \
  --num_bootstrap 10000 \
  --output "${RESULT_DIR}/tuev_relation_screen_3seed_summary.json"
