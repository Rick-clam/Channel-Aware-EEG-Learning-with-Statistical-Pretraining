"""Validate that a run used only validation evidence after fitting."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = json.loads(args.result.read_text(encoding="utf-8"))
    config = record.get("config", {})
    diagnostics = record.get("diagnostics", {})
    if record.get("status") != "completed":
        raise ValueError("result is not completed")
    if config.get("evaluation_mode") != "validation_only":
        raise ValueError("result is not validation_only")
    if config.get("result_scope") != "validation_selection_only":
        raise ValueError("result scope is not validation selection")
    if diagnostics.get("test_set_accessed") is not False:
        raise ValueError("result does not prove test isolation")
    if diagnostics.get("test_loader_constructed") is not False:
        raise ValueError("validation-only run constructed a test loader")
    if "prediction_artifact" in diagnostics:
        raise ValueError("validation-only result contains a test artifact")
    validation_record = diagnostics.get("validation_prediction_artifact")
    if not validation_record:
        raise ValueError("validation prediction artifact is missing")
    prediction_path = Path(validation_record["path"])
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as artifact:
        required = {"logits", "labels", "sample_ids", "subject_ids"}
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"validation artifact misses {sorted(missing)}")
        count = artifact["labels"].shape[0]
        if count == 0:
            raise ValueError("validation artifact is empty")
        if any(artifact[name].shape[0] != count for name in required):
            raise ValueError("validation artifact arrays are misaligned")
        if not np.isfinite(artifact["logits"]).all():
            raise ValueError("validation logits contain non-finite values")
        subjects = int(np.unique(artifact["subject_ids"]).shape[0])

    report = {
        "result": str(args.result),
        "validation_prediction_artifact": str(prediction_path),
        "test_set_accessed": False,
        "test_loader_constructed": False,
        "num_validation_samples": int(count),
        "num_validation_subjects": subjects,
        "finite": True,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
