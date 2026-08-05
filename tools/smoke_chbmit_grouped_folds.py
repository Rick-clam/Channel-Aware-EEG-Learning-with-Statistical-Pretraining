"""Smoke-test CHB-MIT canonical patient folds and cross-directory loading."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_moe
from chbmit_protocol import canonical_chbmit_patient_id


def write_window(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {"X": np.ones((16, 2560), dtype=np.float32), "y": int(label)},
            handle,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "chb"
        examples = (
            ("train", "chb01_01-0.pkl", 0),
            ("train", "chb02_01-0.pkl", 1),
            ("train", "chb03_01-0.pkl", 0),
            ("val", "chb21_01-0.pkl", 1),
            ("val", "chb04_01-0.pkl", 0),
            ("test", "chb05_01-0.pkl", 1),
        )
        for source_split, filename, label in examples:
            write_window(root / source_split / filename, label)
        manifest_path = Path(directory) / "folds.json"
        subprocess.run(
            [
                sys.executable,
                str(project_root / "tools" / "build_chbmit_patient_folds.py"),
                "--root",
                str(root),
                "--num_folds",
                "3",
                "--seed",
                "2026",
                "--output",
                str(manifest_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["canonical_patient_count"] != 5:
            raise AssertionError("chb01/chb21 were not canonicalized")

        fold_index = next(
            fold["fold_index"]
            for fold in manifest["folds"]
            if "chb01" in fold["validation_patients"]
        )
        namespace = SimpleNamespace(
            sampling_rate=200,
            batch_size=2,
            num_workers=0,
            split_manifest_dir=str(Path(directory)),
            allow_subject_overlap=False,
            chbmit_root=str(root),
            chbmit_grouped_manifest=str(manifest_path),
            chbmit_fold_index=fold_index,
        )
        train_loader, test_loader, val_loader = run_moe.prepare_CHB_MIT_dataloader(
            namespace, include_test=False
        )
        if test_loader is not None:
            raise AssertionError("grouped validation constructed a test loader")
        train_patients = {
            canonical_chbmit_patient_id(path)
            for path in train_loader.dataset.files
        }
        validation_patients = {
            canonical_chbmit_patient_id(path)
            for path in val_loader.dataset.files
        }
        if train_patients & validation_patients:
            raise AssertionError("canonical patient leakage across grouped fold")
        chb01_locations = {
            "train" if path in train_loader.dataset.files else "validation"
            for path in ("train/chb01_01-0.pkl", "val/chb21_01-0.pkl")
        }
        if chb01_locations != {"validation"}:
            raise AssertionError("chb01/chb21 were split across partitions")
        try:
            run_moe.prepare_CHB_MIT_dataloader(namespace, include_test=True)
        except ValueError as error:
            if "validation-only" not in str(error):
                raise
        else:
            raise AssertionError("grouped manifest incorrectly allowed final test")

        report = {
            "canonical_patient_count": manifest["canonical_patient_count"],
            "fold_index": fold_index,
            "train_patients": sorted(train_patients),
            "validation_patients": sorted(validation_patients),
            "test_loader_constructed": False,
            "chb01_chb21_same_partition": True,
            "grouped_final_test_blocked": True,
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
