"""Audit whether TUEV processed filenames preserve subject identifiers."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import tuev_subject_id


def summarize_value(value):
    return {
        "type": type(value).__name__,
        "shape": list(value.shape) if hasattr(value, "shape") else None,
        "preview": str(value)[:200],
    }


def inspect_files(directory, limit):
    files = sorted(directory.glob("*.pkl"))
    records = []
    for path in files[:limit]:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        records.append(
            {
                "filename": path.name,
                "stem_tokens": path.stem.split("_"),
                "pickle": {
                    str(key): summarize_value(value)
                    for key, value in sample.items()
                },
            }
        )
    return files, records


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
        "--manifest",
        type=Path,
        default=Path("results/manifests/tuev_subject_split.json"),
    )
    parser.add_argument("--sample_files", type=int, default=12)
    parser.add_argument(
        "--repair_manifest",
        action="store_true",
        help="replace only test_subjects with parsed official eval indices",
    )
    args = parser.parse_args()

    train_files, train_records = inspect_files(
        args.root / "processed_train", args.sample_files
    )
    eval_files, eval_records = inspect_files(
        args.root / "processed_eval", args.sample_files
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolved_eval_subjects = sorted(
        {tuev_subject_id(path.name, "eval") for path in eval_files}
    )
    manifest_repaired = False
    if args.repair_manifest:
        manifest["test_subjects"] = resolved_eval_subjects
        manifest["subject_id_parser_version"] = 2
        args.manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        manifest_repaired = True
    report = {
        "root": str(args.root),
        "counts": {
            "train_files": len(train_files),
            "eval_files": len(eval_files),
        },
        "train_first_token_unique": len(
            {path.stem.split("_")[0] for path in train_files}
        ),
        "eval_first_token_counts": Counter(
            path.stem.split("_")[0] for path in eval_files
        ),
        "eval_first_two_token_unique": len(
            {
                "_".join(path.stem.split("_")[:2])
                for path in eval_files
            }
        ),
        "resolved_subject_counts": {
            "train": len(
                {tuev_subject_id(path.name, "train") for path in train_files}
            ),
            "eval": len(
                {tuev_subject_id(path.name, "eval") for path in eval_files}
            ),
        },
        "resolved_eval_subject_examples": resolved_eval_subjects[:20],
        "manifest_repaired": manifest_repaired,
        "manifest_counts": {
            key: len(value) if isinstance(value, list) else value
            for key, value in manifest.items()
        },
        "manifest_examples": {
            key: value[:20]
            for key, value in manifest.items()
            if isinstance(value, list)
        },
        "train_samples": train_records,
        "eval_samples": eval_records,
        "warning": (
            "A filename token is a valid subject ID only if confirmed against "
            "the original corpus metadata; token uniqueness alone is not proof."
        ),
    }
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
