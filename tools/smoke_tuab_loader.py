"""Build TUAB loaders without reading samples or starting training."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_moe import prepare_TUAB_dataloader


def main() -> None:
    args = SimpleNamespace(
        seed=2026,
        sampling_rate=200,
        batch_size=16,
        num_workers=0,
        tuab_root=(
            "/home/dataset/tuh_eeg/tuh_eeg_abnormal/"
            "tuh_eeg_abnormal/v3.0.1/edf/processed"
        ),
        allow_subject_overlap=False,
        split_manifest_dir="results/manifests",
    )
    train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
    print(
        json.dumps(
            {
                "train_files": len(train_loader.dataset.files),
                "val_files": len(val_loader.dataset.files),
                "test_files": len(test_loader.dataset.files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
