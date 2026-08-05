"""Runtime smoke for held-out routing diagnostic tensors."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate


def main():
    torch.manual_seed(2026)
    model = dwmoespace_newgate(
        6,
        gate_type="input",
        expert_axis="temporal",
        norm_type="local_filter",
        relation_mode="dynamic_normalized",
    )
    details = model.forward_with_expert_details(
        torch.randn(3, 16, 2000)
    )
    report = {
        "shapes": {
            name: list(value.shape) for name, value in details.items()
        },
        "finite": {
            name: bool(torch.isfinite(value).all())
            for name, value in details.items()
        },
        "gate_row_sums": details["gate_weights"].sum(dim=-1).tolist(),
    }
    reconstructed_logits = (
        (
            details["channel_expert_scores"]
            * details["gate_weights"].unsqueeze(-1)
        )
        .sum(dim=2)
        .mean(dim=1)
        + model.classifier[-1].bias
    )
    report["mixture_reconstruction_max_abs_error"] = float(
        (reconstructed_logits - details["logits"]).abs().max()
    )
    print(json.dumps(report, indent=2))
    expected = {
        "logits": [3, 6],
        "expert_logits": [3, 3, 6],
        "gate_weights": [3, 16, 3],
        "channel_expert_scores": [3, 16, 3, 6],
    }
    if report["shapes"] != expected:
        raise SystemExit("Unexpected routing diagnostic shape")
    if not all(report["finite"].values()):
        raise SystemExit("Routing diagnostic contains non-finite values")
    if not torch.allclose(
        details["gate_weights"].sum(dim=-1),
        torch.ones(3, 16),
        atol=1e-6,
    ):
        raise SystemExit("Gate weights are not normalized")
    if not torch.allclose(
        reconstructed_logits, details["logits"], atol=1e-6
    ):
        raise SystemExit("Channel-expert scores do not reconstruct logits")


if __name__ == "__main__":
    main()
