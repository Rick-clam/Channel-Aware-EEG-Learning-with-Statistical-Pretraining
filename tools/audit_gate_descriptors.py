"""Measure gate descriptor variability on real TUEV signals."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate
from utils import TUEVLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuev_root",
        default="/home/dataset/tuh_eeg/tuh_eeg_events/"
        "tuh_eeg_events/v2.0.1/edf",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def descriptors(features):
    flat = features.reshape(features.shape[0], features.shape[1], -1)
    return torch.stack(
        (
            flat.mean(dim=-1),
            flat.std(dim=-1, unbiased=False),
            flat.abs().mean(dim=-1),
        ),
        dim=-1,
    )


def summarize(values):
    # Variance is computed across independent windows for each channel, then
    # summarized across the 16 channels.
    sample_variance_by_channel = values.var(dim=0, unbiased=False)
    return {
        "mean_by_descriptor": values.mean(dim=(0, 1)).tolist(),
        "sample_variance_mean_over_channels": sample_variance_by_channel.mean(
            dim=0
        ).tolist(),
        "sample_variance_min_over_channels": sample_variance_by_channel.min(
            dim=0
        ).values.tolist(),
        "sample_variance_max_over_channels": sample_variance_by_channel.max(
            dim=0
        ).values.tolist(),
    }


def main():
    args = parse_args()
    root = os.path.join(args.tuev_root, "processed_train")
    all_files = sorted(
        filename for filename in os.listdir(root) if filename.endswith(".pkl")
    )
    indices = np.linspace(
        0, len(all_files) - 1, num=min(args.samples, len(all_files)), dtype=int
    )
    files = [all_files[index] for index in indices]
    loader = torch.utils.data.DataLoader(
        TUEVLoader(root, files, sampling_rate=200),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    raw_batches = []
    encoded_batches = {"legacy_global": [], "local_filter": []}
    gate_batches = {"raw": [], "encoded": []}
    attention_batches = []
    models = {
        norm_type: dwmoespace_newgate(
            num_classes=6,
            gate_type="input",
            expert_axis="temporal",
            norm_type=norm_type,
            gate_descriptor_source="raw",
            relation_mode="dynamic_normalized",
        ).eval()
        for norm_type in ("legacy_global", "local_filter")
    }
    encoded_gate_model = dwmoespace_newgate(
        num_classes=6,
        gate_type="input",
        expert_axis="temporal",
        norm_type="local_filter",
        gate_descriptor_source="encoded",
        relation_mode="dynamic_normalized",
    ).eval()
    prior = torch.linspace(
        -0.3,
        0.3,
        steps=models["local_filter"].moe.gate_weights.numel(),
    ).reshape_as(models["local_filter"].moe.gate_weights)
    with torch.no_grad():
        models["local_filter"].moe.gate_weights.copy_(prior)
        encoded_gate_model.moe.gate_weights.copy_(prior)
    nested_static_weights = torch.softmax(prior, dim=-1)

    with torch.no_grad():
        for signals, _labels in loader:
            raw_descriptor_batch = descriptors(signals)
            raw_batches.append(raw_descriptor_batch)
            attention_batches.append(
                models["local_filter"].moe.experts[0].channel_attn(
                    raw_descriptor_batch
                )
            )
            for norm_type, model in models.items():
                grid = model.moe.temporal_conv(signals, return_grid=True)
                encoded_batches[norm_type].append(descriptors(grid))
            models["local_filter"](signals)
            gate_batches["raw"].append(
                models["local_filter"].moe.last_gate_weights
            )
            encoded_gate_model(signals)
            gate_batches["encoded"].append(
                encoded_gate_model.moe.last_gate_weights
            )

    raw = torch.cat(raw_batches)
    encoded = {
        key: torch.cat(value) for key, value in encoded_batches.items()
    }
    gates = {key: torch.cat(value) for key, value in gate_batches.items()}
    report = {
        "dataset": "TUEV processed_train",
        "sample_count": len(files),
        "descriptor_order": ["mean", "std", "mean_absolute"],
        "raw_signal_descriptors": summarize(raw),
        "encoded_descriptors": {
            key: summarize(value) for key, value in encoded.items()
        },
        "untrained_gate_output_sample_variance": {
            key: value.var(dim=0, unbiased=False).mean(dim=0).tolist()
            for key, value in gates.items()
        },
        "nested_input_gate_max_abs_error_vs_static_prior": float(
            max(
                (
                    gates["raw"]
                    - nested_static_weights.unsqueeze(0)
                ).abs().max(),
                (
                    gates["encoded"]
                    - nested_static_weights.unsqueeze(0)
                ).abs().max(),
            )
        ),
        "raw_descriptor_attention_sample_variance_mean": float(
            torch.cat(attention_batches)
            .var(dim=0, unbiased=False)
            .mean()
        ),
        "interpretation": (
            "Descriptor variance is a precondition check only. Routing evidence "
            "requires trained full-validation/test diagnostics."
        ),
    }
    print(json.dumps(report, indent=2))
    raw_variance = report["raw_signal_descriptors"][
        "sample_variance_min_over_channels"
    ]
    if any(value <= 1e-10 for value in raw_variance):
        raise SystemExit("A raw gate descriptor is effectively constant")
    if report["nested_input_gate_max_abs_error_vs_static_prior"] > 1e-6:
        raise SystemExit("Input gate does not initialize to static submodel")
    if report["raw_descriptor_attention_sample_variance_mean"] <= 1e-12:
        raise SystemExit("Raw-descriptor attention is effectively constant")


if __name__ == "__main__":
    main()
