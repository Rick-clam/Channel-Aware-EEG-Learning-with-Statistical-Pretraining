"""Validate one TUEV main pretrained validation/final-test result artifact."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prediction_artifact(record, key):
    artifact = record["diagnostics"][key]
    path = Path(artifact["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if artifact.get("sha256") != sha256(path):
        raise RuntimeError(f"{key} sha256 no longer matches")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"logits", "labels", "sample_ids", "subject_ids"}
        missing = required - set(arrays.files)
        if missing:
            raise RuntimeError(f"{key} misses arrays: {sorted(missing)}")
        logits = arrays["logits"]
        labels = arrays["labels"]
        subjects = arrays["subject_ids"]
        if logits.ndim != 2 or logits.shape[1] != 6:
            raise RuntimeError(f"unexpected TUEV logits shape: {logits.shape}")
        if labels.shape != (logits.shape[0],):
            raise RuntimeError("TUEV labels are not sample-aligned")
        if subjects.shape != (logits.shape[0],):
            raise RuntimeError("TUEV subject IDs are not sample-aligned")
        if logits.shape[0] == 0:
            raise RuntimeError("TUEV prediction artifact is empty")
        if not np.isfinite(logits).all():
            raise RuntimeError("TUEV logits contain non-finite values")
    if "num_samples" in artifact and int(artifact["num_samples"]) != int(
        logits.shape[0]
    ):
        raise RuntimeError(f"{key} num_samples no longer matches")
    return str(path), int(logits.shape[0]), int(len(np.unique(subjects)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--transfer_mode", default="full_finetune")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode", choices=["validation_only", "final_test"], required=True
    )
    args = parser.parse_args()

    record = json.loads(args.result.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError("TUEV result is not completed")
    config = record["config"]
    expected = {
        "dataset": "TUEV",
        "model": "dwmoespace_newgate",
        "n_classes": 6,
        "feature_out": 299,
        "expert_axis": "temporal",
        "expert_kernels": "7",
        "gate_type": "uniform",
        "relation_mode": "none",
        "norm_type": "local_filter",
        "transfer_mode": args.transfer_mode,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "evaluation_mode": args.mode,
    }
    mismatches = {
        key: {"expected": value, "recorded": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "TUEV main result configuration mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not config.get("pretrain_model_path"):
        raise RuntimeError("TUEV main pretrained result lacks a checkpoint")

    diagnostics = record["diagnostics"]
    if args.mode == "validation_only":
        if diagnostics.get("test_loader_constructed") is not False:
            raise RuntimeError("TUEV validation result constructed test loader")
        if diagnostics.get("test_set_accessed") is not False:
            raise RuntimeError("TUEV validation result accessed test set")
        artifact_path, sample_count, subject_count = validate_prediction_artifact(
            record, "validation_prediction_artifact"
        )
    else:
        if diagnostics.get("test_loader_constructed") is not True:
            raise RuntimeError("TUEV final result did not construct test loader")
        if diagnostics.get("test_set_accessed") is not True:
            raise RuntimeError("TUEV final result did not access test set")
        if diagnostics.get("evaluation_lock", {}).get("mode") != "final_test":
            raise RuntimeError("TUEV final result lacks final-test lock")
        artifact_path, sample_count, subject_count = validate_prediction_artifact(
            record, "prediction_artifact"
        )

    metrics = record["metrics"]
    required = (
        ("val_cohen", "val_acc", "val_f1", "balanced_accuracy")
        if args.mode == "validation_only"
        else ("test_cohen", "test_acc", "test_f1", "balanced_accuracy")
    )
    missing_metrics = [
        key
        for key in required
        if key not in metrics
        or not isinstance(metrics[key], (int, float))
        or not math.isfinite(metrics[key])
    ]
    if missing_metrics:
        raise RuntimeError(f"TUEV result lacks finite metrics: {missing_metrics}")
    checkpoint = diagnostics.get("best_checkpoint", {})
    if not Path(checkpoint.get("path", "")).is_file():
        raise RuntimeError("TUEV result lacks its best checkpoint")
    snapshot = diagnostics.get("source_snapshot_at_start", {})
    if not snapshot.get("source_snapshot_sha256"):
        raise RuntimeError("TUEV result lacks source snapshot")

    print(
        json.dumps(
            {
                "validated": True,
                "result": str(args.result),
                "mode": args.mode,
                "seed": args.seed,
                "pretrain_model_path": config["pretrain_model_path"],
                "transfer_mode": args.transfer_mode,
                "label_fraction": args.label_fraction,
                "sample_count": sample_count,
                "subject_count": subject_count,
                "prediction_artifact": artifact_path,
                "metrics": {key: metrics[key] for key in required},
                "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
