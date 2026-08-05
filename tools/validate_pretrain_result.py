"""Validate one fixed-step pretraining result before downstream transfer."""

import argparse
import hashlib
import json
import math
from pathlib import Path


BACKBONES = {
    "dense_original": {
    "feature_out": 100,
    "gate_type": "static",
    "expert_axis": "legacy",
    "norm_type": "auto",
    "relation_mode": "dynamic_normalized",
    "expert_kernels": "3,7,15",
    "sampling_rate": 200,
    "window_seconds": 5,
    "need_token": 20,
    },
    "tuev_original": {
        "feature_out": 299,
        "gate_type": "uniform",
        "expert_axis": "temporal",
        "norm_type": "local_filter",
        "relation_mode": "none",
        "expert_kernels": "7",
        "sampling_rate": 200,
        "window_seconds": 5,
        "need_token": 20,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--logdataset", default="tuabchbmit")
    parser.add_argument("--held_out_dataset", default="tuev")
    parser.add_argument(
        "--backbone",
        choices=sorted(BACKBONES),
        default="dense_original",
        help="Expected original architecture family for this pretraining run.",
    )
    args = parser.parse_args()
    record = json.loads(args.result.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError("pretraining result is not completed")
    config = record["config"]
    expected = {
        **BACKBONES[args.backbone],
        "objective": args.objective,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "logdataset": args.logdataset,
        "held_out_dataset": args.held_out_dataset,
        "source_sampling": "balanced",
    }
    mismatches = {
        key: {"expected": value, "recorded": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "pretraining configuration mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    diagnostics = record["diagnostics"]
    checkpoint_record = diagnostics["selected_checkpoint"]
    if not isinstance(checkpoint_record, dict):
        raise RuntimeError(
            "pretraining checkpoint must be recorded with path/SHA-256/size"
        )
    checkpoint = Path(checkpoint_record["path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    current_checkpoint_record = {
        "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "size_bytes": checkpoint.stat().st_size,
    }
    if current_checkpoint_record != checkpoint_record:
        raise RuntimeError("pretraining checkpoint metadata no longer matches")
    if diagnostics.get("selection_rule") != (
        f"fixed final optimizer step {args.max_steps}"
    ):
        raise RuntimeError("unexpected pretraining checkpoint selection rule")
    if diagnostics.get("held_out_dataset") != args.held_out_dataset:
        raise RuntimeError(
            f"{args.held_out_dataset} is not the held-out pretraining dataset"
        )
    if args.objective in (
        "normalized_masked_stats",
        "shuffled_masked_stats",
    ) and not diagnostics.get("stat_normalization"):
        raise RuntimeError("masked statistic result lacks normalization artifact")
    metrics = record.get("metrics", {})
    finite_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }
    if not finite_metrics:
        raise RuntimeError("pretraining result has no finite metric")
    provenance = record.get("provenance", {})
    if not provenance.get("source_snapshot_sha256"):
        raise RuntimeError("pretraining result lacks a source snapshot")
    print(
        json.dumps(
            {
                "result": str(args.result),
                "objective": args.objective,
                "seed": args.seed,
                "checkpoint": str(checkpoint),
                "source_snapshot_sha256": provenance[
                    "source_snapshot_sha256"
                ],
                "finite_metrics": finite_metrics,
                "validated": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
