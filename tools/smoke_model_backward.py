"""Check gradient coverage for every revised relation variant."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate


def run_case(relation_mode: str, gate_type: str):
    torch.manual_seed(2026)
    model = dwmoespace_newgate(
        num_classes=2,
        gate_type=gate_type,
        expert_axis="temporal",
        relation_mode=relation_mode,
    )
    model.train()
    output = model(torch.randn(2, 16, 2000))
    output.square().mean().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    kernels = [
        list(expert.temporal_encoder[0].kernel_size)
        for expert in model.moe.experts
    ]
    return {
        "relation_mode": relation_mode,
        "gate_type": gate_type,
        "output_shape": list(output.shape),
        "temporal_kernel_shapes": kernels,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "missing_gradients": missing,
        "nonfinite_gradients": nonfinite,
    }


def main():
    relation_modes = (
        "none",
        "attention",
        "static",
        "dynamic",
        "dynamic_normalized",
        "spatial1x1",
        "dynamic_normalized_attention",
    )
    records = [run_case(mode, "input") for mode in relation_modes]
    records.extend(
        [
            run_case("dynamic_normalized", "uniform"),
            run_case("dynamic_normalized", "static"),
        ]
    )
    print(json.dumps(records, indent=2))
    if any(
        record["missing_gradients"] or record["nonfinite_gradients"]
        for record in records
    ):
        raise SystemExit(1)
    if any(
        any(shape[0] != 1 for shape in record["temporal_kernel_shapes"])
        for record in records
    ):
        raise SystemExit("A revised expert kernel spans the EEG-channel axis")


if __name__ == "__main__":
    main()
