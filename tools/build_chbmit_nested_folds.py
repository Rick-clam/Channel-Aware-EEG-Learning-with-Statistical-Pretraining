"""Derive nested patient-grouped CHB-MIT folds from the audited v2 folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL = (
    "nested patient-grouped train-validation-test evaluation over the complete "
    "legacy directory tree"
)


def metadata(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source.get("manifest_version") != 2:
        raise ValueError("source must be the audited version-2 grouped manifest")
    if source.get("dataset") != "CHB_MIT":
        raise ValueError("source manifest is not CHB_MIT")
    source_folds = sorted(source["folds"], key=lambda fold: fold["fold_index"])
    num_folds = int(source["num_folds"])
    if len(source_folds) != num_folds or num_folds < 3:
        raise ValueError("invalid source fold inventory")

    patient_records = {
        record["patient_id"]: record for record in source["patients"]
    }
    known_patients = set(patient_records)

    def totals(patients: set[str]) -> tuple[int, int]:
        return (
            sum(patient_records[p]["window_count"] for p in patients),
            sum(patient_records[p]["positive_count"] for p in patients),
        )

    nested_folds = []
    seen_test_patients: set[str] = set()
    for outer_index in range(num_folds):
        test_patients = set(source_folds[outer_index]["validation_patients"])
        inner_index = (outer_index + 1) % num_folds
        validation_patients = set(
            source_folds[inner_index]["validation_patients"]
        )
        train_patients = known_patients - test_patients - validation_patients
        if not train_patients or not validation_patients or not test_patients:
            raise ValueError(f"empty nested partition in outer fold {outer_index}")
        if (
            train_patients & validation_patients
            or train_patients & test_patients
            or validation_patients & test_patients
        ):
            raise AssertionError("patient overlap in nested fold")
        if train_patients | validation_patients | test_patients != known_patients:
            raise AssertionError("nested fold does not cover every patient")
        seen_test_patients.update(test_patients)
        train_windows, train_positives = totals(train_patients)
        val_windows, val_positives = totals(validation_patients)
        test_windows, test_positives = totals(test_patients)
        nested_folds.append(
            {
                "fold_index": outer_index,
                "inner_validation_source_fold": inner_index,
                "train_patients": sorted(train_patients),
                "validation_patients": sorted(validation_patients),
                "test_patients": sorted(test_patients),
                "train_window_count": train_windows,
                "validation_window_count": val_windows,
                "test_window_count": test_windows,
                "train_positive_count": train_positives,
                "validation_positive_count": val_positives,
                "test_positive_count": test_positives,
            }
        )
    if seen_test_patients != known_patients:
        raise AssertionError("each canonical patient must appear in an outer test")

    report = {
        **{
            key: source[key]
            for key in (
                "dataset",
                "root",
                "canonical_case_to_patient",
                "source_splits",
                "source_split_window_counts",
                "total_window_count",
                "relative_path_inventory_sha256",
                "canonical_patient_count",
                "fold_seed",
                "num_folds",
                "patients",
            )
        },
        "manifest_version": 3,
        "protocol": PROTOCOL,
        "source_grouped_manifest": metadata(args.source),
        "inner_validation_rule": (
            "For outer test fold i, use audited source fold (i+1) modulo "
            "num_folds as validation and all remaining patients for training."
        ),
        "folds": nested_folds,
        "notes": [
            "Every outer test patient is absent from training and validation.",
            "Every inner validation patient is absent from training and test.",
            "Legacy directory names are storage locations only.",
            "chb21 remains canonicalized to chb01 before all assignments.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest_version": report["manifest_version"],
        "num_folds": report["num_folds"],
        "canonical_patient_count": report["canonical_patient_count"],
        "source_manifest_sha256": report["source_grouped_manifest"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
