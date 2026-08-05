"""Audit one TUAB/CHB-MIT validation or locked-test result artifact."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_npz_artifact(record, key):
    artifact = record["diagnostics"][key]
    path = Path(artifact["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    current_path = str(path)
    current_sha = sha256(path)
    if artifact.get("path") != current_path:
        raise ValueError(f"{key} path no longer matches")
    if artifact.get("sha256") != current_sha:
        raise ValueError(f"{key} sha256 no longer matches")
    if "size_bytes" in artifact and artifact["size_bytes"] != path.stat().st_size:
        raise ValueError(f"{key} size no longer matches")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"logits", "labels", "sample_ids", "subject_ids"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{key} misses arrays: {sorted(missing)}")
        count = arrays["labels"].shape[0]
        if count == 0:
            raise ValueError(f"{key} is empty")
        for name in required:
            if arrays[name].shape[0] != count:
                raise ValueError(f"{key}.{name} is not sample-aligned")
        if not np.isfinite(arrays["logits"]).all():
            raise ValueError(f"{key} logits contain non-finite values")
    if "num_samples" in artifact and int(artifact["num_samples"]) != int(count):
        raise ValueError(f"{key} num_samples no longer matches")
    return str(path), int(count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    record = json.loads(args.result.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError("result is not completed")
    config = record["config"]
    diagnostics = record["diagnostics"]
    if config.get("dataset") not in {"TUAB", "CHB_MIT"}:
        raise ValueError("result is not TUAB/CHB-MIT")
    if config.get("model") != "dwmoespace_newgate":
        raise ValueError("result is not the dense pretraining-transfer model")
    if config.get("expert_axis") != "legacy":
        raise ValueError("unexpected expert axis")
    if config.get("norm_type") != "auto":
        raise ValueError("unexpected normalization")
    if config.get("relation_mode") != "dynamic_normalized":
        raise ValueError("unexpected relation mode")
    if config.get("expert_kernels") != "3,7,15":
        raise ValueError("unexpected expert kernels")
    if config.get("feature_out") != 100:
        raise ValueError("unexpected expert output width")
    if config.get("gate_type") != "static":
        raise ValueError("unexpected gate type")

    mode = config["evaluation_mode"]
    if mode == "validation_only":
        if diagnostics.get("test_set_accessed") is not False:
            raise ValueError("validation result accessed the test set")
        if diagnostics.get("test_loader_constructed") is not False:
            raise ValueError("validation result constructed a test loader")
        artifact_path, sample_count = validate_npz_artifact(
            record, "validation_prediction_artifact"
        )
    elif mode == "final_test":
        if diagnostics.get("test_set_accessed") is not True:
            raise ValueError("final result did not access the test set")
        if diagnostics.get("test_loader_constructed") is not True:
            raise ValueError("final result did not construct the test loader")
        if diagnostics.get("evaluation_lock", {}).get("mode") != "final_test":
            raise ValueError("final result lacks a verified final-test lock")
        artifact_path, sample_count = validate_npz_artifact(
            record, "prediction_artifact"
        )
    else:
        raise ValueError(f"unsupported evaluation mode: {mode}")

    snapshot = diagnostics.get("source_snapshot_at_start", {})
    if not snapshot.get("source_snapshot_sha256"):
        raise ValueError("missing executable source snapshot")

    print(
        json.dumps(
            {
                "status": "passed",
                "result": str(args.result),
                "dataset": config["dataset"],
                "mode": mode,
                "seed": config["seed"],
                "pretrain_model_path": config.get("pretrain_model_path", ""),
                "transfer_mode": config.get("transfer_mode"),
                "sample_count": sample_count,
                "prediction_artifact": artifact_path,
                "metrics": record["metrics"],
                "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
