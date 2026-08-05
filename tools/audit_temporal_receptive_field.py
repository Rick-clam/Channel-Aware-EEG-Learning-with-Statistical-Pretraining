"""Empirically audit temporal receptive fields with exact autograd support."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate


SAMPLE_RATE_HZ = 200
EPSILON = 1e-12


def summarize_gradient(gradient):
    active = torch.nonzero(gradient.abs() > EPSILON, as_tuple=False).flatten()
    if active.numel() == 0:
        raise RuntimeError("No non-zero input gradient found")
    first = int(active.min())
    last = int(active.max())
    span = last - first + 1
    return {
        "first_sample": first,
        "last_sample": last,
        "nonzero_sample_count": int(active.numel()),
        "support_span_samples": span,
        "support_span_ms_at_200hz": 1000.0 * span / SAMPLE_RATE_HZ,
    }


def revised_expert_support(raw_length):
    torch.manual_seed(2026)
    model = dwmoespace_newgate(
        num_classes=2,
        gate_type="uniform",
        expert_axis="temporal",
        relation_mode="dynamic_normalized",
    )
    model.eval()
    signal = torch.randn(1, 16, raw_length, requires_grad=True)
    grid = model.moe.temporal_conv(signal, return_grid=True)
    expert_input = grid.permute(0, 2, 1, 3)
    records = []
    for expert in model.moe.experts:
        if signal.grad is not None:
            signal.grad.zero_()
        encoded = expert.temporal_encoder(expert_input)
        encoded[0, 0, 8, 40].backward(retain_graph=True)
        support = summarize_gradient(signal.grad[0, 8])
        support.update(
            {
                "expert_kernel_latent_steps": expert.temporal_kernel,
                "kernel_shape": list(expert.temporal_encoder[0].kernel_size),
            }
        )
        records.append(support)
    return records


def legacy_normalization_support(raw_length):
    torch.manual_seed(2026)
    model = dwmoespace_newgate(
        num_classes=2,
        gate_type="uniform",
        expert_axis="legacy",
    )
    model.eval()
    signal = torch.randn(1, 16, raw_length, requires_grad=True)
    grid = model.moe.temporal_conv(signal, return_grid=True)
    grid[0, 0, 8, 40].backward()
    result = summarize_gradient(signal.grad[0, 8])
    result["normalization"] = "GroupNorm(1,C), retained only for legacy baseline"
    return result


def main():
    input_lengths = (1000, 2000)
    report = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "gradient_threshold": EPSILON,
        "by_input_length": {
            str(raw_length): {
                "physical_window_seconds": raw_length / SAMPLE_RATE_HZ,
                "revised_local_normalization": revised_expert_support(
                    raw_length
                ),
                "legacy_global_normalization": (
                    legacy_normalization_support(raw_length)
                ),
            }
            for raw_length in input_lengths
        },
        "interpretation": (
            "Revised kernel spans are local for a central latent position. "
            "The relation matrix subsequently pools these features over latent "
            "time and is therefore sample-global by design."
        ),
    }
    print(json.dumps(report, indent=2))
    for raw_length in input_lengths:
        length_report = report["by_input_length"][str(raw_length)]
        revised = length_report["revised_local_normalization"]
        if not (
            revised[0]["support_span_samples"]
            < revised[1]["support_span_samples"]
            < revised[2]["support_span_samples"]
        ):
            raise SystemExit(
                f"Temporal receptive fields are not ordered for {raw_length}"
            )
        if (
            length_report["legacy_global_normalization"][
                "support_span_samples"
            ]
            != raw_length
        ):
            raise SystemExit(
                "Legacy normalization was expected to make support global"
            )


if __name__ == "__main__":
    main()
