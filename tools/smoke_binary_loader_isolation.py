"""Prove validation-only binary loaders never enumerate test directories."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_moe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    common = {
        "sampling_rate": 200,
        "batch_size": 128,
        "num_workers": 0,
        "split_manifest_dir": "results/manifests",
        "allow_subject_overlap": False,
        "tuab_root": (
            "/home/dataset/tuh_eeg/tuh_eeg_abnormal/"
            "tuh_eeg_abnormal/v3.0.1/edf/processed"
        ),
        "chbmit_root": "/home/dataset/CHB-MIT/clean_segments",
        "chbmit_grouped_manifest": "",
        "chbmit_fold_index": -1,
    }
    original_listdir = run_moe.os.listdir
    touched = []

    def guarded_listdir(path):
        normalized = os.path.normpath(path)
        touched.append(normalized)
        if os.path.basename(normalized) == "test":
            raise AssertionError(
                f"validation-only loader enumerated test directory: {path}"
            )
        return original_listdir(path)

    run_moe.os.listdir = guarded_listdir
    report = {}
    try:
        for dataset, prepare in (("TUAB", run_moe.prepare_TUAB_dataloader),):
            namespace = SimpleNamespace(**common)
            train_loader, test_loader, val_loader = prepare(
                namespace, include_test=False
            )
            if test_loader is not None:
                raise AssertionError(f"{dataset} constructed a test loader")
            report[dataset] = {
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(val_loader.dataset),
                "test_loader_constructed": False,
                "effective_split_manifest": (
                    namespace.effective_split_manifest
                ),
            }
        namespace = SimpleNamespace(**common)
        try:
            run_moe.prepare_CHB_MIT_dataloader(namespace, include_test=False)
        except RuntimeError as error:
            report["CHB_MIT_legacy_split"] = {
                "blocked": True,
                "reason": str(error),
                "required_repair": (
                    "use a frozen patient-grouped manifest after canonicalizing "
                    "chb21 to chb01"
                ),
            }
        else:
            raise AssertionError(
                "legacy CHB-MIT split was not rejected after patient canonicalization"
            )
    finally:
        run_moe.os.listdir = original_listdir

    report["enumerated_directories"] = touched
    report["test_directory_enumerated"] = False
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
