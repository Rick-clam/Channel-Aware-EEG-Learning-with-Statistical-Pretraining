"""Build nested, class-stratified 1%/10% TUEV training subsets."""

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_utils import artifact_metadata
from utils import tuev_subject_id


FRACTIONS = (0.01, 0.1)


def inventory_sha256(files):
    digest = hashlib.sha256()
    for filename in sorted(files):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_label(path):
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    label = int(sample["label"][0] - 1)
    if label not in range(6):
        raise ValueError(f"Unexpected TUEV label {label} in {path}")
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuev_root",
        type=Path,
        default=Path(
            "/home/dataset/tuh_eeg/tuh_eeg_events/"
            "tuh_eeg_events/v2.0.1/edf"
        ),
    )
    parser.add_argument(
        "--tuev_split_manifest",
        type=Path,
        default=Path("results/manifests/tuev_subject_split.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(
        args.tuev_split_manifest.read_text(encoding="utf-8")
    )
    if split.get("subject_id_parser_version") != 2:
        raise ValueError("TUEV split manifest uses an obsolete subject parser")
    train_subjects = set(split["train_subjects"])
    validation_subjects = set(split["val_subjects"])
    if train_subjects & validation_subjects:
        raise RuntimeError("TUEV split manifest has subject overlap")
    source_dir = args.tuev_root / "processed_train"
    all_files = sorted(path.name for path in source_dir.glob("*.pkl"))
    train_files = [
        filename
        for filename in all_files
        if tuev_subject_id(filename, "train") in train_subjects
    ]
    if len(train_files) + sum(
        tuev_subject_id(filename, "train") in validation_subjects
        for filename in all_files
    ) != len(all_files):
        raise RuntimeError("TUEV split manifest does not cover processed_train")

    by_label = defaultdict(list)
    for index, filename in enumerate(train_files, start=1):
        by_label[read_label(source_dir / filename)].append(filename)
        if index % 10000 == 0:
            print(f"read labels for {index}/{len(train_files)} files", flush=True)
    if set(by_label) != set(range(6)):
        raise RuntimeError(f"Missing TUEV classes: {sorted(set(range(6)) - set(by_label))}")
    shuffled = {}
    for label, files in sorted(by_label.items()):
        ordered = sorted(files)
        random.Random(args.seed + 1009 * label).shuffle(ordered)
        shuffled[label] = ordered

    subsets = {}
    for fraction in FRACTIONS:
        selected = []
        class_counts = {}
        for label, files in sorted(shuffled.items()):
            count = max(1, int(len(files) * fraction))
            selected.extend(files[:count])
            class_counts[str(label)] = count
        selected.sort()
        subsets[f"{fraction:g}"] = {
            "file_count": len(selected),
            "class_counts": class_counts,
            "files": selected,
        }
    one_percent = set(subsets["0.01"]["files"])
    ten_percent = set(subsets["0.1"]["files"])
    if not one_percent.issubset(ten_percent):
        raise AssertionError("1% TUEV subset is not nested within 10%")

    report = {
        "artifact_version": 1,
        "dataset": "TUEV",
        "protocol": "nested class-stratified fractions of frozen training subjects",
        "seed": args.seed,
        "tuev_root": str(args.tuev_root),
        "tuev_split_manifest": str(args.tuev_split_manifest),
        "tuev_split_manifest_sha256": artifact_metadata(
            args.tuev_split_manifest
        )["sha256"],
        "full_train_file_count": len(train_files),
        "full_train_class_counts": {
            str(label): len(files) for label, files in sorted(by_label.items())
        },
        "full_train_file_inventory_sha256": inventory_sha256(train_files),
        "fractions": list(FRACTIONS),
        "subsets": subsets,
        "test_or_processed_eval_accessed": False,
        "note": (
            "Each class is deterministically shuffled once; the first floor(n*f) "
            "files (minimum one) form each fraction, so 1% is nested in 10%."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "full_train_file_count": report["full_train_file_count"],
                "full_train_class_counts": report["full_train_class_counts"],
                "subset_counts": {
                    key: value["class_counts"]
                    for key, value in subsets.items()
                },
                "nested": True,
                "test_or_processed_eval_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
