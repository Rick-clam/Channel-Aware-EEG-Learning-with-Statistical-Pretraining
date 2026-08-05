"""Rebuild CHB-MIT fold assignments from an audited patient-count manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chbmit_protocol import enumerate_chbmit_windows, relative_inventory_sha256
from tools.build_chbmit_patient_folds import assign_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    root = Path(source["root"])
    relative_paths = enumerate_chbmit_windows(root)
    if len(relative_paths) != source.get("total_window_count"):
        raise RuntimeError("CHB-MIT total window count differs from source manifest")
    if relative_inventory_sha256(relative_paths) != source.get(
        "relative_path_inventory_sha256"
    ):
        raise RuntimeError("CHB-MIT path inventory differs from source manifest")

    patient_records = source["patients"]
    all_patients = {record["patient_id"] for record in patient_records}
    lookup = {record["patient_id"]: record for record in patient_records}
    assigned = assign_folds(patient_records, args.num_folds, args.seed)
    folds = []
    seen_validation = set()
    for index, fold in enumerate(assigned):
        validation_patients = sorted(fold["patients"])
        training_patients = sorted(all_patients - set(validation_patients))
        if not validation_patients:
            raise RuntimeError(f"fold {index} is empty")
        if set(training_patients) & set(validation_patients):
            raise RuntimeError("canonical patient overlap in grouped fold")
        seen_validation.update(validation_patients)
        folds.append(
            {
                "fold_index": index,
                "train_patients": training_patients,
                "validation_patients": validation_patients,
                "train_window_count": sum(
                    lookup[patient]["window_count"]
                    for patient in training_patients
                ),
                "validation_window_count": fold["window_count"],
                "train_positive_count": sum(
                    lookup[patient]["positive_count"]
                    for patient in training_patients
                ),
                "validation_positive_count": fold["positive_count"],
            }
        )
    if seen_validation != all_patients:
        raise RuntimeError("each canonical patient must validate exactly once")

    report = dict(source)
    report["manifest_version"] = 2
    report["fold_seed"] = args.seed
    report["num_folds"] = args.num_folds
    report["folds"] = folds
    report["rebalanced_from"] = str(args.input)
    report["fold_assignment_rule"] = (
        "500 deterministic equal-patient-count random restarts with pair-swap "
        "local search minimizing normalized window and positive-count error"
    )
    report["notes"] = list(source.get("notes", [])) + [
        "Manifest v2 forbids empty validation folds by construction."
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_version": 2,
                "folds": [
                    {
                        "fold_index": fold["fold_index"],
                        "validation_patients": fold["validation_patients"],
                        "validation_window_count": fold[
                            "validation_window_count"
                        ],
                        "validation_positive_count": fold[
                            "validation_positive_count"
                        ],
                    }
                    for fold in folds
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
