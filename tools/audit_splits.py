"""Create dataset split manifests and detect filename/subject overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set


def subject_id(filename: str, separator: str) -> str:
    return Path(filename).name.split(separator)[0]


def list_files(root: Path, split: str) -> List[str]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    return sorted(
        str(path.relative_to(root)) for path in split_dir.iterdir() if path.is_file()
    )


def intersections(values: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    names = list(values)
    output: Dict[str, List[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            output[f"{left}__{right}"] = sorted(values[left] & values[right])
    return output


def manifest_hash(files: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files):
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--subject-separator", default="_")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = {split: list_files(args.root, split) for split in args.splits}
    file_sets = {split: set(items) for split, items in files.items()}
    subjects = {
        split: {subject_id(item, args.subject_separator) for item in items}
        for split, items in files.items()
    }
    file_overlap = intersections(file_sets)
    subject_overlap = intersections(subjects)
    record = {
        "dataset": args.dataset,
        "root": str(args.root.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "splits": {
            split: {
                "file_count": len(items),
                "subject_count": len(subjects[split]),
                "sha256": manifest_hash(items),
                "files": items,
            }
            for split, items in files.items()
        },
        "file_overlap": file_overlap,
        "subject_overlap": subject_overlap,
        "passed": not any(file_overlap.values()) and not any(subject_overlap.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary = {
        "dataset": record["dataset"],
        "root": record["root"],
        "split_counts": {
            split: {
                "files": details["file_count"],
                "subjects": details["subject_count"],
            }
            for split, details in record["splits"].items()
        },
        "file_overlap_counts": {
            key: len(value) for key, value in file_overlap.items()
        },
        "subject_overlap_counts": {
            key: len(value) for key, value in subject_overlap.items()
        },
        "passed": record["passed"],
    }
    print(json.dumps(summary, indent=2))
    if not record["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
