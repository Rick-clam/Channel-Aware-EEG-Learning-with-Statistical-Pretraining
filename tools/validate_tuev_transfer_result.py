"""Validate one TUEV validation-only transfer result and prediction artifact."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--transfer_mode", required=True)
    parser.add_argument("--label_fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    record = json.loads(args.result.read_text(encoding="utf-8"))
    config = record["config"]
    expected = {
        "dataset": "TUEV",
        "model": "dwmoespace_newgate",
        "feature_out": 299,
        "expert_axis": "temporal",
        "expert_kernels": "7",
        "gate_type": "uniform",
        "relation_mode": "none",
        "norm_type": "local_filter",
        "transfer_mode": args.transfer_mode,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "evaluation_mode": "validation_only",
    }
    mismatches = {
        key: {"expected": value, "recorded": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "TUEV transfer configuration mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if args.variant == "scratch":
        if config.get("pretrain_model_path"):
            raise RuntimeError("scratch result unexpectedly loads pretraining")
    elif not config.get("pretrain_model_path"):
        raise RuntimeError("pretrained transfer result lacks a checkpoint")
    diagnostics = record["diagnostics"]
    if diagnostics.get("test_loader_constructed"):
        raise RuntimeError("TUEV validation result constructed a test loader")
    if diagnostics.get("test_set_accessed"):
        raise RuntimeError("TUEV validation result accessed the test set")
    prediction = diagnostics["validation_prediction_artifact"]
    prediction_path = Path(prediction["path"])
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    arrays = np.load(prediction_path, allow_pickle=False)
    logits = arrays["logits"]
    labels = arrays["labels"]
    subjects = arrays["subject_ids"]
    if logits.shape != (6744, 6) or labels.shape != (6744,):
        raise RuntimeError(
            f"unexpected TUEV validation prediction shapes: {logits.shape}, {labels.shape}"
        )
    if len(np.unique(subjects)) != 29:
        raise RuntimeError("unexpected TUEV validation subject count")
    if not np.isfinite(logits).all():
        raise RuntimeError("TUEV validation logits contain non-finite values")
    metrics = record["metrics"]
    required_metrics = ("val_cohen", "val_acc", "val_f1", "balanced_accuracy")
    if any(
        key not in metrics
        or not isinstance(metrics[key], (int, float))
        or not math.isfinite(metrics[key])
        for key in required_metrics
    ):
        raise RuntimeError("TUEV transfer result lacks finite metrics")
    checkpoint = diagnostics.get("best_checkpoint", {})
    if not Path(checkpoint.get("path", "")).is_file():
        raise RuntimeError("TUEV transfer result lacks its best checkpoint")
    print(
        json.dumps(
            {
                "result": str(args.result),
                "variant": args.variant,
                "transfer_mode": args.transfer_mode,
                "label_fraction": args.label_fraction,
                "seed": args.seed,
                "metrics": {key: metrics[key] for key in required_metrics},
                "validation_windows": int(labels.shape[0]),
                "validation_subjects": int(len(np.unique(subjects))),
                "test_loader_constructed": False,
                "test_set_accessed": False,
                "validated": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
