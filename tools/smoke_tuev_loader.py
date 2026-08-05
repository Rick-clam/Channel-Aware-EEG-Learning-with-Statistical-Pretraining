"""Build TUEV loaders without reading samples or starting training."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_multi_moe import prepare_TUEV_dataloader


def main() -> None:
    args = SimpleNamespace(
        seed=2026,
        split_seed=2026,
        sampling_rate=200,
        batch_size=16,
        num_workers=0,
        tuev_root=(
            "/home/dataset/tuh_eeg/tuh_eeg_events/"
            "tuh_eeg_events/v2.0.1/edf"
        ),
        tuev_split_manifest="results/manifests/tuev_subject_split.json",
    )
    train_loader, test_loader, val_loader = prepare_TUEV_dataloader(
        args, include_test=True
    )
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
