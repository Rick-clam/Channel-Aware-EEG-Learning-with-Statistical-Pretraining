"""Prove a validation-only TUEV loader never enumerates processed_eval."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_multi_moe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
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
    original_listdir = run_multi_moe.os.listdir
    touched = []

    def guarded_listdir(path):
        normalized = os.path.normpath(path)
        touched.append(normalized)
        if os.path.basename(normalized) == "processed_eval":
            raise AssertionError(
                "validation-only TUEV loader enumerated processed_eval"
            )
        return original_listdir(path)

    run_multi_moe.os.listdir = guarded_listdir
    try:
        train_loader, test_loader, val_loader = (
            run_multi_moe.prepare_TUEV_dataloader(
                args, include_test=False
            )
        )
    finally:
        run_multi_moe.os.listdir = original_listdir
    if test_loader is not None:
        raise AssertionError("validation-only TUEV constructed a test loader")
    report = {
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_loader_constructed": False,
        "enumerated_directories": touched,
        "processed_eval_enumerated": False,
    }
    rendered = json.dumps(report, indent=2)
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
