"""Audit a complete CHB-MIT grouped-fold manifest without opening EEG files."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chbmit_protocol import (
    canonical_chbmit_patient_id,
    enumerate_chbmit_windows,
    relative_inventory_sha256,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 2:
        raise ValueError("unsupported manifest version")
    root = Path(manifest["root"])
    relative_paths = enumerate_chbmit_windows(root)
    if len(relative_paths) != manifest.get("total_window_count"):
        raise RuntimeError("CHB-MIT total window count differs from manifest")
    if relative_inventory_sha256(relative_paths) != manifest.get(
        "relative_path_inventory_sha256"
    ):
        raise RuntimeError("CHB-MIT path inventory differs from manifest")
    current_patient_counts = Counter(
        canonical_chbmit_patient_id(path) for path in relative_paths
    )
    recorded_patients = {
        record["patient_id"]: record for record in manifest["patients"]
    }
    if set(current_patient_counts) != set(recorded_patients):
        raise RuntimeError("canonical patient set differs from manifest")
    for patient, count in current_patient_counts.items():
        if recorded_patients[patient]["window_count"] != count:
            raise RuntimeError(f"window count differs for {patient}")

    all_patients = set(recorded_patients)
    validation_seen = Counter()
    fold_reports = []
    for fold in manifest["folds"]:
        train = set(fold["train_patients"])
        validation = set(fold["validation_patients"])
        if train & validation:
            raise RuntimeError("patient overlap in grouped fold")
        if train | validation != all_patients:
            raise RuntimeError("grouped fold does not cover every patient")
        validation_seen.update(validation)
        expected_train = sum(current_patient_counts[p] for p in train)
        expected_validation = sum(
            current_patient_counts[p] for p in validation
        )
        if expected_validation == 0:
            raise RuntimeError("grouped manifest contains an empty validation fold")
        if expected_train != fold["train_window_count"]:
            raise RuntimeError("fold training window count mismatch")
        if expected_validation != fold["validation_window_count"]:
            raise RuntimeError("fold validation window count mismatch")
        fold_reports.append(
            {
                "fold_index": fold["fold_index"],
                "train_patient_count": len(train),
                "validation_patient_count": len(validation),
                "train_window_count": expected_train,
                "validation_window_count": expected_validation,
                "train_positive_count": fold["train_positive_count"],
                "validation_positive_count": fold[
                    "validation_positive_count"
                ],
                "validation_prevalence": (
                    fold["validation_positive_count"]
                    / expected_validation
                ),
            }
        )
    if set(validation_seen) != all_patients or any(
        count != 1 for count in validation_seen.values()
    ):
        raise RuntimeError("each patient must validate exactly once")
    chb01_cases = recorded_patients["chb01"]["case_ids"]
    if not {"chb01", "chb21"}.issubset(set(chb01_cases)):
        raise RuntimeError("chb01/chb21 canonical merge is absent")

    report = {
        "manifest": str(args.manifest),
        "root": str(root),
        "total_window_count": len(relative_paths),
        "canonical_patient_count": len(all_patients),
        "chb01_case_ids": chb01_cases,
        "chb01_chb21_merged": True,
        "zero_patient_overlap_all_folds": True,
        "each_patient_validates_once": True,
        "test_loader_constructed": False,
        "folds": fold_reports,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
