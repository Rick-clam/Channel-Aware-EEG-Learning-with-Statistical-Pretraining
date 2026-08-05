"""Validate the corrected TUEV smoke result and prediction export."""

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
        raise ValueError("smoke result is not completed")
    if config.get("evaluation_mode") != "smoke_test":
        raise ValueError("artifact is not a smoke_test result")
    if config.get("result_scope") != "non_evidentiary_smoke":
        raise ValueError("smoke result is not marked non-evidentiary")
    if diagnostics.get("test_set_accessed") is not True:
        raise ValueError("smoke result does not declare test access")
    if diagnostics.get("test_loader_constructed") is not True:
        raise ValueError("smoke result did not construct the limited test loader")

    prediction_path = Path(
        diagnostics["prediction_artifact"]["path"]
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as artifact:
        required = {
            "logits",
            "labels",
            "sample_ids",
            "subject_ids",
            "expert_logits",
            "gate_weights",
            "channel_expert_scores",
            "permuted_gate_logits",
            "gate_permutation",
        }
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"prediction artifact misses {sorted(missing)}")
        arrays = {name: artifact[name] for name in required}

    count = arrays["labels"].shape[0]
    if count == 0:
        raise ValueError("prediction artifact is empty")
    if any(values.shape[0] != count for values in arrays.values()):
        raise ValueError("prediction arrays have inconsistent sample counts")
    parsed_ids = np.asarray(
        [Path(name).name.split("_")[1] for name in arrays["sample_ids"]]
    )
    if not np.array_equal(parsed_ids, arrays["subject_ids"]):
        raise ValueError("exported TUEV subject IDs do not match filenames")
    if np.any(np.char.str_len(arrays["subject_ids"].astype(str)) == 0):
        raise ValueError("empty TUEV subject ID")
    if arrays["gate_weights"].ndim != 3:
        raise ValueError("gate_weights must be [sample,channel,expert]")
    if arrays["channel_expert_scores"].ndim != 4:
        raise ValueError(
            "channel_expert_scores must be [sample,channel,expert,class]"
        )
    if not all(np.isfinite(values).all() for values in (
        arrays["logits"],
        arrays["expert_logits"],
        arrays["gate_weights"],
        arrays["channel_expert_scores"],
        arrays["permuted_gate_logits"],
    )):
        raise ValueError("prediction artifact contains non-finite values")
    permutation = arrays["gate_permutation"]
    if not np.array_equal(np.sort(permutation), np.arange(count)):
        raise ValueError("gate permutation is not a bijection")
    if count > 1 and np.any(permutation == np.arange(count)):
        raise ValueError("gate permutation contains fixed points")

    report = {
        "result": str(args.result),
        "prediction_artifact": str(prediction_path),
        "num_samples": int(count),
        "num_subjects_in_limited_smoke": int(
            np.unique(arrays["subject_ids"]).shape[0]
        ),
        "logits_shape": list(arrays["logits"].shape),
        "expert_logits_shape": list(arrays["expert_logits"].shape),
        "gate_weights_shape": list(arrays["gate_weights"].shape),
        "channel_expert_scores_shape": list(
            arrays["channel_expert_scores"].shape
        ),
        "subject_id_alignment": True,
        "finite": True,
        "permutation_bijection_no_fixed_points": True,
        "paper_evidence": False,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
