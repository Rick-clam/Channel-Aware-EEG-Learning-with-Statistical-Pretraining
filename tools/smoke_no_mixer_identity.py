"""Verify that the no-mixer relation control is exactly the identity path."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import TemporalChannelRelationExpert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(2026)
    expert = TemporalChannelRelationExpert(
        output_dim=100,
        temporal_kernel=7,
        relation_mode="none",
    )
    features = torch.randn(2, 16, 25, 80, requires_grad=True)
    descriptors = torch.randn(2, 16, 3)
    actual = expert(features, descriptors)
    flat = features.reshape(2, 16, 25 * 80)
    expected = expert.feature_projection(flat)
    max_error = float((actual - expected).abs().max().item())
    if max_error != 0.0:
        raise ValueError(f"no-mixer path is not identity: {max_error}")
    actual.square().mean().backward()
    missing = [
        name
        for name, parameter in expert.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise ValueError(f"trainable no-mixer parameters lack gradients: {missing}")
    report = {
        "input_shape": list(features.shape),
        "output_shape": list(actual.shape),
        "identity_max_abs_error": max_error,
        "missing_trainable_gradients": missing,
        "finite": bool(torch.isfinite(actual).all().item()),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
