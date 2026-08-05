"""Exhaustively join processed TUEV samples to raw EDF subject directories."""

import argparse
import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import tuev_subject_id


def source_stem(processed_path):
    stem = processed_path.stem
    if "-" not in stem:
        raise ValueError(f"Processed filename lacks event suffix: {stem}")
    return stem.rsplit("-", 1)[0]


def raw_index(directory):
    index = {}
    for path in sorted(directory.glob("*/*.edf")):
        if path.stem in index:
            raise ValueError(f"Duplicate raw EDF stem: {path.stem}")
        index[path.stem] = path
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/dataset/tuh_eeg/tuh_eeg_events/"
            "tuh_eeg_events/v2.0.1/edf"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/manifests/tuev_processed_file_mapping.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/manifests/tuev_processed_file_mapping_summary.json"
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts = {}
    subject_counts = {}
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for partition, processed_name, raw_name in (
            ("train", "processed_train", "train"),
            ("eval", "processed_eval", "eval"),
        ):
            raw_by_stem = raw_index(args.root / raw_name)
            processed_files = sorted(
                (args.root / processed_name).glob("*.pkl")
            )
            resolved_subjects = set()
            for processed_path in processed_files:
                stem = source_stem(processed_path)
                raw_path = raw_by_stem.get(stem)
                if raw_path is None:
                    raise ValueError(
                        f"No raw EDF for {processed_path.name} ({stem})"
                    )
                parsed_subject = tuev_subject_id(
                    processed_path.name, partition
                )
                raw_subject = raw_path.parent.name
                if parsed_subject != raw_subject:
                    raise ValueError(
                        f"Subject mismatch for {processed_path.name}: "
                        f"parser={parsed_subject}, raw={raw_subject}"
                    )
                resolved_subjects.add(parsed_subject)
                record = {
                    "partition": partition,
                    "processed_file": processed_path.name,
                    "raw_edf": str(raw_path.relative_to(args.root)),
                    "subject_id": parsed_subject,
                    "processed_size_bytes": processed_path.stat().st_size,
                    "raw_size_bytes": raw_path.stat().st_size,
                }
                line = json.dumps(
                    record, sort_keys=True, separators=(",", ":")
                )
                handle.write(line + "\n")
                digest.update(line.encode("utf-8"))
                digest.update(b"\n")
            counts[partition] = len(processed_files)
            subject_counts[partition] = len(resolved_subjects)

    summary = {
        "dataset": "TUEV v2.0.1",
        "subject_id_parser_version": 2,
        "mapping_path": str(args.output),
        "mapping_sha256": digest.hexdigest(),
        "processed_file_counts": counts,
        "subject_counts": subject_counts,
        "join_invariant": (
            "Every processed filename source stem maps to exactly one raw EDF "
            "and parsed subject_id equals that EDF's parent directory."
        ),
        "content_hash_scope": (
            "The mapping hash covers filenames, raw paths, subject IDs, and "
            "file sizes, not full EEG payload bytes."
        ),
    }
    args.summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
