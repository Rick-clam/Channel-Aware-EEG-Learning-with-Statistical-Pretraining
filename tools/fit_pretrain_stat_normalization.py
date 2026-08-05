"""Fit deterministic source-training-only mean/std for pretraining targets."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import TwoDatasetPretrainLoader, collate_fn_two_dataset_pretrain


DATASET_PAIRS = {
    "tuevchbmit": ("tuev", "chbmit"),
    "tuabchbmit": ("tuab", "chbmit"),
    "tuabtuev": ("tuab", "tuev"),
}


def compute_statistics(batch, segment_count):
    if batch.shape[-1] % segment_count:
        raise ValueError("window samples must be divisible by segment_count")
    segmented = batch.reshape(
        batch.shape[0], batch.shape[1], segment_count, -1
    )
    mean = segmented.mean(dim=-1)
    std = segmented.std(dim=-1, unbiased=True)
    safe_std = std.unsqueeze(-1).clamp_min(1e-6)
    skew = (
        (segmented - mean.unsqueeze(-1)) / safe_std
    ).pow(3).mean(dim=-1)
    return torch.stack((mean, std, skew), dim=2)


def inventory_digest(loader, datasets):
    roots = {
        "tuab": loader.base.root_tuab,
        "tuev": loader.base.root_tuev,
        "chbmit": loader.base.root_chbmit,
    }
    paths = {
        "tuab": loader.base.tuab_list,
        "tuev": loader.base.tuev_list,
        "chbmit": loader.base.chbmit_list,
    }
    digest = hashlib.sha256()
    counts = {}
    for dataset in datasets:
        relative = sorted(
            os.path.relpath(path, roots[dataset]).replace(os.sep, "/")
            for path in paths[dataset]
        )
        counts[dataset] = len(relative)
        for path in relative:
            digest.update(dataset.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest(), counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdataset", choices=sorted(DATASET_PAIRS), required=True)
    parser.add_argument("--samples_per_dataset", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sampling_rate", type=int, default=200)
    parser.add_argument("--window_seconds", type=int, default=5)
    parser.add_argument("--segment_count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tuab_root",
        default="/home/dataset/tuh_eeg/tuh_eeg_abnormal/"
        "tuh_eeg_abnormal/v3.0.1/edf/processed",
    )
    parser.add_argument(
        "--tuev_root",
        default="/home/dataset/tuh_eeg/tuh_eeg_events/"
        "tuh_eeg_events/v2.0.1/edf",
    )
    parser.add_argument(
        "--chbmit_root",
        default="/home/dataset/CHB-MIT/clean_segments",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_dataset <= 0:
        raise ValueError("samples_per_dataset must be positive")
    loader = TwoDatasetPretrainLoader(
        args.tuab_root,
        args.tuev_root,
        args.chbmit_root,
        DATASET_PAIRS[args.logdataset],
        target_sample_rate=args.sampling_rate,
        target_window_seconds=args.window_seconds,
    )
    offsets = {}
    running = 0
    selected_indices = []
    selected_counts = {}
    for dataset in loader.datasets:
        offsets[dataset] = running
        length = loader.lengths[dataset]
        count = min(args.samples_per_dataset, length)
        selected_counts[dataset] = count
        selected_indices.extend(
            running + (index * length // count) for index in range(count)
        )
        running += length

    subset = torch.utils.data.Subset(loader, selected_indices)
    data_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn_two_dataset_pretrain,
    )
    sums = torch.zeros(3, dtype=torch.float64)
    sum_squares = torch.zeros(3, dtype=torch.float64)
    value_count = 0
    sample_count = 0
    for batch in data_loader:
        stats = compute_statistics(batch, args.segment_count).double()
        sums += stats.sum(dim=(0, 1, 3))
        sum_squares += stats.square().sum(dim=(0, 1, 3))
        value_count += stats.shape[0] * stats.shape[1] * stats.shape[3]
        sample_count += stats.shape[0]
    mean = sums / value_count
    variance = (sum_squares / value_count - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("non-finite fitted statistic normalization")
    if torch.any(std <= 0):
        raise RuntimeError("zero-variance fitted statistic normalization")

    inventory_sha, base_record_counts = inventory_digest(
        loader, loader.datasets
    )
    report = {
        "artifact_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "logdataset": args.logdataset,
        "source_datasets": list(loader.datasets),
        "held_out_dataset": (
            {"tuevchbmit": "tuab", "tuabchbmit": "tuev", "tuabtuev": "chbmit"}
            [args.logdataset]
        ),
        "source_splits": {
            "tuab": ["train"],
            "tuev": ["processed_train"],
            "chbmit": ["train"],
        },
        "sampling_rate_hz": args.sampling_rate,
        "window_seconds": args.window_seconds,
        "window_samples": args.sampling_rate * args.window_seconds,
        "segment_count": args.segment_count,
        "stats_features": ["mean", "std", "skewness"],
        "selection_rule": (
            "per source dataset, take min(samples_per_dataset, virtual_length) "
            "evenly spaced virtual indices in deterministic dataset order"
        ),
        "requested_samples_per_dataset": args.samples_per_dataset,
        "selected_samples_per_dataset": selected_counts,
        "fitted_sample_count": sample_count,
        "values_per_statistic": value_count,
        "base_source_record_counts": base_record_counts,
        "source_relative_path_inventory_sha256": inventory_sha,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
