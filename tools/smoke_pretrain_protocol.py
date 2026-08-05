"""Validate harmonized pretraining windows and the direct-statistics head."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import UnsupervisedPretrain
from utils import TwoDatasetPretrainLoader


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    loader = TwoDatasetPretrainLoader(
        args.tuab_root,
        args.tuev_root,
        args.chbmit_root,
        ("tuev", "chbmit"),
        target_sample_rate=200,
        target_window_seconds=5,
    )
    offsets = {}
    running = 0
    for name in loader.datasets:
        offsets[name] = running
        running += loader.lengths[name]

    samples = []
    source_records = []
    for name in loader.datasets:
        sample, source_flag = loader[offsets[name]]
        samples.append(sample)
        source_records.append(
            {
                "dataset": name,
                "virtual_samples": loader.lengths[name],
                "base_records": loader.base_lengths[name],
                "segments_per_record": loader.segments_per_record[name],
                "shape": list(sample.shape),
                "source_flag": source_flag,
            }
        )

    all_source_paths = (
        loader.base.tuev_list
        + loader.base.tuab_list
        + loader.base.chbmit_list
    )
    forbidden = [
        path
        for path in all_source_paths
        if "/val/" in path
        or "/test/" in path
        or "/processed_eval/" in path
    ]

    batch = torch.stack(samples)
    model = UnsupervisedPretrain(
        raw_time=1000,
        need_token=20,
        expert_output_dim=299,
        gate_type="uniform",
        expert_axis="temporal",
        relation_mode="none",
        norm_type="local_filter",
        expert_kernels="7",
        objective="unmasked_stats",
    )
    raw_stats, predicted_stats = model(batch)
    per_statistic_loss = (predicted_stats - raw_stats).square().mean(
        dim=(0, 1, 3)
    )
    per_statistic_loss.mean().backward()
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]

    report = {
        "sampling_rate_hz": 200,
        "window_seconds": 5,
        "window_samples": 1000,
        "segment_count": 20,
        "segment_samples": 50,
        "segment_duration_seconds": 0.25,
        "objective": "unmasked_stats",
        "backbone": {
            "expert_kernels": [7],
            "expert_output_dim": 299,
            "gate_type": "uniform",
            "relation_mode": "none",
            "norm_type": "local_filter",
        },
        "sources": source_records,
        "forbidden_source_split_paths": forbidden[:20],
        "batch_shape": list(batch.shape),
        "raw_stats_shape": list(raw_stats.shape),
        "predicted_stats_shape": list(predicted_stats.shape),
        "per_statistic_mse": per_statistic_loss.detach().tolist(),
        "finite_loss": bool(torch.isfinite(per_statistic_loss).all()),
        "missing_gradients": missing_gradients,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")

    if forbidden:
        raise SystemExit("A source validation/test path entered pretraining")
    if batch.shape != (2, 16, 1000):
        raise SystemExit(f"Unexpected harmonized batch shape: {batch.shape}")
    if raw_stats.shape != (2, 16, 3, 20):
        raise SystemExit(f"Unexpected target shape: {raw_stats.shape}")
    if not report["finite_loss"] or missing_gradients:
        raise SystemExit("Pretraining forward/backward invariant failed")


if __name__ == "__main__":
    main()
