"""Validate the corrected TUEV manifest through the training entry point."""

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_multi_moe import prepare_TUEV_dataloader
from utils import tuev_subject_id


def main():
    args = SimpleNamespace(
        tuev_root=(
            "/home/dataset/tuh_eeg/tuh_eeg_events/"
            "tuh_eeg_events/v2.0.1/edf"
        ),
        tuev_split_manifest="results/manifests/tuev_subject_split.json",
        split_seed=2026,
        sampling_rate=200,
        batch_size=16,
        num_workers=0,
    )
    loader_stdout = StringIO()
    with redirect_stdout(loader_stdout):
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(
            args, include_test=True
        )
    train_subjects = {
        tuev_subject_id(name, "train")
        for name in train_loader.dataset.files
    }
    val_subjects = {
        tuev_subject_id(name, "train")
        for name in val_loader.dataset.files
    }
    test_subjects = {
        tuev_subject_id(name, "eval")
        for name in test_loader.dataset.files
    }
    report = {
        "window_counts": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "subject_counts": {
            "train": len(train_subjects),
            "validation": len(val_subjects),
            "test": len(test_subjects),
        },
        "overlap_counts": {
            "train_validation": len(train_subjects & val_subjects),
            "train_test": len(train_subjects & test_subjects),
            "validation_test": len(val_subjects & test_subjects),
        },
        "test_subject_examples": sorted(test_subjects)[:20],
        "loader_log": loader_stdout.getvalue().splitlines(),
    }
    print(json.dumps(report, indent=2))
    if any(report["overlap_counts"].values()):
        raise SystemExit("TUEV subject overlap detected")
    if report["subject_counts"] != {
        "train": 261,
        "validation": 29,
        "test": 80,
    }:
        raise SystemExit("Unexpected TUEV subject counts")


if __name__ == "__main__":
    main()
