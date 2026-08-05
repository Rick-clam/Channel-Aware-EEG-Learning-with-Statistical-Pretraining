"""Build deterministic patient-grouped CHB-MIT validation folds."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chbmit_protocol import (
    CASE_TO_CANONICAL_PATIENT,
    SOURCE_SPLITS,
    canonical_chbmit_patient_id,
    chbmit_case_id,
    enumerate_chbmit_windows,
    relative_inventory_sha256,
)


def read_binary_label(path: Path) -> int:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    label = int(sample["y"])
    if label not in (0, 1):
        raise ValueError(f"Non-binary label {label!r} in {path}")
    return label


def assign_folds(patient_records, num_folds, seed):
    """Balance windows/positives with deterministic equal-size local search."""
    if len(patient_records) < num_folds:
        raise ValueError("num_folds exceeds the number of canonical patients")
    total_windows = sum(record["window_count"] for record in patient_records)
    total_positives = sum(record["positive_count"] for record in patient_records)
    target_windows = total_windows / num_folds
    target_positives = total_positives / num_folds

    def score(fold_records):
        total = 0.0
        for fold in fold_records:
            windows = sum(record["window_count"] for record in fold)
            positives = sum(record["positive_count"] for record in fold)
            total += ((windows - target_windows) / target_windows) ** 2
            if target_positives:
                total += ((positives - target_positives) / target_positives) ** 2
        return total

    rng = random.Random(seed)
    best_score = None
    best_assignment = None
    # Equal-size round-robin initialization prevents empty/tiny folds. Pair
    # swaps preserve those patient counts while optimizing both window count
    # and positive count. Fixed restarts and seed make the manifest exact and
    # reproducible.
    for _restart in range(500):
        shuffled = list(patient_records)
        rng.shuffle(shuffled)
        fold_records = [[] for _ in range(num_folds)]
        for index, record in enumerate(shuffled):
            fold_records[index % num_folds].append(record)
        current_score = score(fold_records)
        while True:
            swap = None
            next_score = current_score
            for first_fold in range(num_folds):
                for second_fold in range(first_fold + 1, num_folds):
                    for first_index in range(len(fold_records[first_fold])):
                        for second_index in range(len(fold_records[second_fold])):
                            fold_records[first_fold][first_index], fold_records[
                                second_fold
                            ][second_index] = (
                                fold_records[second_fold][second_index],
                                fold_records[first_fold][first_index],
                            )
                            candidate_score = score(fold_records)
                            fold_records[first_fold][first_index], fold_records[
                                second_fold
                            ][second_index] = (
                                fold_records[second_fold][second_index],
                                fold_records[first_fold][first_index],
                            )
                            if candidate_score < next_score - 1e-12:
                                next_score = candidate_score
                                swap = (
                                    first_fold,
                                    second_fold,
                                    first_index,
                                    second_index,
                                )
            if swap is None:
                break
            first_fold, second_fold, first_index, second_index = swap
            fold_records[first_fold][first_index], fold_records[second_fold][
                second_index
            ] = (
                fold_records[second_fold][second_index],
                fold_records[first_fold][first_index],
            )
            current_score = next_score
        if best_score is None or current_score < best_score:
            best_score = current_score
            best_assignment = [list(fold) for fold in fold_records]

    return [
        {
            "patients": [record["patient_id"] for record in fold],
            "window_count": sum(record["window_count"] for record in fold),
            "positive_count": sum(record["positive_count"] for record in fold),
        }
        for fold in best_assignment
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/dataset/CHB-MIT/clean_segments")
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    relative_paths = enumerate_chbmit_windows(args.root)
    patients = defaultdict(
        lambda: {
            "cases": set(),
            "source_splits": Counter(),
            "window_count": 0,
            "positive_count": 0,
        }
    )
    source_counts = Counter()
    for index, relative_path in enumerate(relative_paths, start=1):
        source_split, filename = relative_path.split("/", 1)
        patient_id = canonical_chbmit_patient_id(filename)
        case_id = chbmit_case_id(filename)
        label = read_binary_label(args.root / relative_path)
        record = patients[patient_id]
        record["cases"].add(case_id)
        record["source_splits"][source_split] += 1
        record["window_count"] += 1
        record["positive_count"] += label
        source_counts[source_split] += 1
        if index % 25000 == 0:
            print(f"scanned {index}/{len(relative_paths)} windows", flush=True)

    patient_records = []
    for patient_id, record in sorted(patients.items()):
        patient_records.append(
            {
                "patient_id": patient_id,
                "case_ids": sorted(record["cases"]),
                "source_split_counts": {
                    split: record["source_splits"].get(split, 0)
                    for split in SOURCE_SPLITS
                },
                "window_count": record["window_count"],
                "positive_count": record["positive_count"],
                "negative_count": record["window_count"]
                - record["positive_count"],
            }
        )
    assigned = assign_folds(patient_records, args.num_folds, args.seed)
    all_patients = {record["patient_id"] for record in patient_records}
    folds = []
    seen_validation = set()
    lookup = {record["patient_id"]: record for record in patient_records}
    for index, fold in enumerate(assigned):
        validation_patients = sorted(fold["patients"])
        training_patients = sorted(all_patients - set(validation_patients))
        if set(training_patients) & set(validation_patients):
            raise AssertionError("canonical patient overlap in a grouped fold")
        seen_validation.update(validation_patients)
        folds.append(
            {
                "fold_index": index,
                "train_patients": training_patients,
                "validation_patients": validation_patients,
                "train_window_count": sum(
                    lookup[patient]["window_count"] for patient in training_patients
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
        raise AssertionError("each canonical patient must validate exactly once")

    report = {
        "manifest_version": 2,
        "dataset": "CHB_MIT",
        "protocol": "patient-grouped cross-validation over the complete legacy directory tree",
        "root": str(args.root),
        "canonical_case_to_patient": CASE_TO_CANONICAL_PATIENT,
        "source_splits": list(SOURCE_SPLITS),
        "source_split_window_counts": dict(sorted(source_counts.items())),
        "total_window_count": len(relative_paths),
        "relative_path_inventory_sha256": relative_inventory_sha256(relative_paths),
        "canonical_patient_count": len(patient_records),
        "fold_seed": args.seed,
        "num_folds": args.num_folds,
        "patients": patient_records,
        "folds": folds,
        "notes": [
            "chb21 is canonicalized to chb01 before fold assignment.",
            "Legacy train/val/test directory names are source locations only and do not define the new folds.",
            "Architecture screening is validation-only; final confirmation requires a separately frozen LOPO/nested protocol.",
            "Manifest v2 uses deterministic equal-patient-count local search, so empty/tiny validation folds are prevented.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "total_window_count",
        "canonical_patient_count",
        "num_folds",
        "relative_path_inventory_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
