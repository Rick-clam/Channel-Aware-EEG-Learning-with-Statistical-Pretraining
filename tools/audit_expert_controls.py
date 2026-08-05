"""Audit expert-set controls and select a parameter-matched wide expert."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate


def build(kernels, output_dim, relation_mode="none"):
    return dwmoespace_newgate(
        num_classes=6,
        expert_output_dim=output_dim,
        gate_type="uniform",
        expert_axis="temporal",
        norm_type="local_filter",
        relation_mode=relation_mode,
        expert_kernels=kernels,
    )


def trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def profile_case(name, kernels, output_dim, relation_mode, input_samples):
    torch.manual_seed(2026)
    model = build(kernels, output_dim, relation_mode=relation_mode)
    sample = torch.randn(1, 16, input_samples)
    model.eval()
    with profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            output = model(sample)
    flops = sum(event.flops for event in prof.key_averages())
    model.train()
    loss = model(sample).square().mean()
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    return {
        "name": name,
        "kernels": list(kernels),
        "expert_output_dim": output_dim,
        "trainable_parameters": trainable_parameters(model),
        "profiler_flops_batch1": int(flops),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "missing_gradients": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relation_mode", default="none", choices=("none", "dynamic_normalized")
    )
    parser.add_argument("--input_samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    target = build((3, 7, 15), 100, relation_mode=args.relation_mode)
    target_parameters = trainable_parameters(target)
    width_search = []
    for output_dim in range(100, 801):
        parameters = trainable_parameters(
            build((7,), output_dim, relation_mode=args.relation_mode)
        )
        width_search.append(
            {
                "expert_output_dim": output_dim,
                "trainable_parameters": parameters,
                "absolute_parameter_delta": abs(
                    parameters - target_parameters
                ),
            }
        )
    closest = min(
        width_search, key=lambda record: record["absolute_parameter_delta"]
    )
    cases = [
        profile_case(
            "multi_scale_3_7_15", (3, 7, 15), 100,
            args.relation_mode, args.input_samples,
        ),
        profile_case(
            "single_7", (7,), 100, args.relation_mode, args.input_samples,
        ),
        profile_case(
            "duplicate_7_7_7", (7, 7, 7), 100,
            args.relation_mode, args.input_samples,
        ),
        profile_case(
            "wide_single_7_parameter_matched",
            (7,),
            closest["expert_output_dim"],
            args.relation_mode,
            args.input_samples,
        ),
    ]
    for record in cases:
        record["parameter_delta_vs_multi_scale"] = (
            record["trainable_parameters"] - target_parameters
        )
        record["parameter_ratio_vs_multi_scale"] = (
            record["trainable_parameters"] / target_parameters
        )
        record["flop_ratio_vs_multi_scale"] = (
            record["profiler_flops_batch1"]
            / cases[0]["profiler_flops_batch1"]
        )
    report = {
        "relation_mode": args.relation_mode,
        "input_samples": args.input_samples,
        "target_trainable_parameters": target_parameters,
        "selected_wide_single": closest,
        "cases": cases,
        "note": (
            "Profiler FLOPs cover supported PyTorch operators. The wide "
            "single expert is selected only by trainable-parameter distance; "
            "its FLOP ratio is reported separately."
        ),
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if any(record["missing_gradients"] for record in cases):
        raise SystemExit("An expert control has missing gradients")
    if not all(record["finite"] for record in cases):
        raise SystemExit("An expert control produced non-finite output")


if __name__ == "__main__":
    main()
