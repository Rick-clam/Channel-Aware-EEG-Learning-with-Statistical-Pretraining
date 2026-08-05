"""Audit parameter/compute matching and norm-axis factorial availability."""

import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate


CASES = (
    ("dynamic_normalized", False),
    ("dynamic", False),
    ("attention", False),
    ("none", False),
    ("none", True),
    ("static", False),
    ("static", True),
    ("spatial1x1", False),
    ("spatial1x1", True),
    ("static_conditioned_matched", False),
    ("spatial1x1_conditioned_matched", False),
)


def audit_case(relation_mode, match_dynamic_budget):
    torch.manual_seed(2026)
    model = dwmoespace_newgate(
        num_classes=6,
        gate_type="uniform",
        expert_axis="temporal",
        norm_type="local_filter",
        relation_mode=relation_mode,
        match_dynamic_budget=match_dynamic_budget,
    )
    model.eval()
    sample = torch.randn(1, 16, 2000)
    with profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            output = model(sample)
    flops = sum(event.flops for event in prof.key_averages())

    model.train()
    output = model(sample)
    output.square().mean().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    return {
        "relation_mode": relation_mode,
        "match_dynamic_budget": match_dynamic_budget,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "profiler_flops_batch1": int(flops),
        "output_shape": list(output.shape),
        "missing_gradients": missing,
    }


def factorial_cases():
    records = []
    for expert_axis in ("legacy", "temporal"):
        for norm_type in ("legacy_global", "local_filter"):
            model = dwmoespace_newgate(
                num_classes=6,
                gate_type="uniform",
                expert_axis=expert_axis,
                norm_type=norm_type,
                relation_mode="dynamic_normalized",
            )
            with torch.no_grad():
                output = model(torch.randn(1, 16, 2000))
            records.append(
                {
                    "expert_axis": expert_axis,
                    "norm_type": norm_type,
                    "resolved_norm_type": model.moe.norm_type,
                    "output_shape": list(output.shape),
                    "finite": bool(torch.isfinite(output).all()),
                }
            )
    return records


def main():
    controls = [audit_case(*case) for case in CASES]
    target = controls[0]
    for record in controls:
        record["parameter_delta_vs_dynamic"] = (
            record["trainable_parameters"] - target["trainable_parameters"]
        )
        record["flop_ratio_vs_dynamic"] = (
            record["profiler_flops_batch1"]
            / target["profiler_flops_batch1"]
        )
    report = {
        "controls": controls,
        "axis_norm_factorial": factorial_cases(),
        "note": (
            "Profiler FLOPs cover supported PyTorch operators. Controls are "
            "called matched only when both parameter and FLOP deltas are small."
        ),
    }
    print(json.dumps(report, indent=2))
    if any(record["missing_gradients"] for record in controls):
        raise SystemExit("A control has trainable parameters without gradients")
    if not all(record["finite"] for record in report["axis_norm_factorial"]):
        raise SystemExit("A norm-axis factorial forward pass failed")


if __name__ == "__main__":
    main()
