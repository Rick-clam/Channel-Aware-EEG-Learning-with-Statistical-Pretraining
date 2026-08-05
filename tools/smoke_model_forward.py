"""CPU-safe forward smoke test for all expert aggregation rules."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate


def main() -> None:
    torch.manual_seed(2026)
    sample = torch.randn(1, 16, 2000)
    records = []
    for expert_axis in ("legacy", "temporal"):
        relation_modes = (
            ("dynamic_normalized",)
            if expert_axis == "legacy"
            else (
                "none",
                "attention",
                "static",
                "dynamic",
                "dynamic_normalized",
                "spatial1x1",
                "dynamic_normalized_attention",
            )
        )
        for relation_mode in relation_modes:
            for gate_type in ("uniform", "static", "input"):
                model = dwmoespace_newgate(
                    num_classes=2,
                    gate_type=gate_type,
                    expert_axis=expert_axis,
                    relation_mode=relation_mode,
                )
                model.eval()
                with torch.no_grad():
                    output = model(sample)
                weights = model.moe.last_gate_weights
                relation = getattr(
                    model.moe.experts[0], "last_relation_matrix", None
                )
                record = {
                    "expert_axis": expert_axis,
                    "relation_mode": relation_mode,
                    "gate_type": gate_type,
                    "output_shape": list(output.shape),
                    "gate_shape": list(weights.shape),
                    "first_channel_weights": weights[0, 0].tolist(),
                    "finite": bool(torch.isfinite(output).all()),
                    "gate_sums_to_one": bool(
                        torch.allclose(
                            weights.sum(dim=-1),
                            torch.ones_like(weights.sum(dim=-1)),
                            atol=1e-6,
                        )
                    ),
                }
                if relation is not None:
                    record["relation_shape"] = list(relation.shape)
                    record["relation_rows_sum_to_one"] = bool(
                        torch.allclose(
                            relation.sum(dim=-1),
                            torch.ones_like(relation.sum(dim=-1)),
                            atol=1e-6,
                        )
                    )
                records.append(record)
    print(json.dumps(records, indent=2))
    if not all(record["finite"] and record["gate_sums_to_one"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
