"""Audit backbone/decoder parameter budgets for pretraining controls."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import UnsupervisedPretrain


OBJECTIVES = UnsupervisedPretrain.SUPPORTED_OBJECTIVES


def trainable(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for objective in OBJECTIVES:
        torch.manual_seed(2026)
        model = UnsupervisedPretrain(
            expert_output_dim=299,
            raw_time=1000,
            need_token=20,
            gate_type="uniform",
            expert_axis="temporal",
            relation_mode="none",
            norm_type="local_filter",
            expert_kernels="7",
            objective=objective,
        )
        sample = torch.randn(1, 16, 1000)
        encoder_input = sample.clone()
        encoder_input[:, :, :500] = 0
        model.eval()
        with profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=True,
        ) as profiler:
            with torch.no_grad():
                target, prediction = model(
                    sample,
                    sample if objective == "unmasked_stats" else encoder_input,
                )
        encoder_parameters = trainable(model.moe)
        total_parameters = trainable(model)
        records.append(
            {
                "objective": objective,
                "trainable_total_parameters": total_parameters,
                "trainable_encoder_parameters": encoder_parameters,
                "trainable_decoder_parameters": total_parameters
                - encoder_parameters,
                "encoder_initialization_sha256": state_digest(model.moe),
                "profiler_flops_batch1": int(
                    sum(event.flops for event in profiler.key_averages())
                ),
                "target_shape": list(target.shape),
                "prediction_shape": list(prediction.shape),
                "finite": bool(
                    torch.isfinite(target).all()
                    and torch.isfinite(prediction).all()
                ),
            }
        )
    encoder_counts = {
        record["trainable_encoder_parameters"] for record in records
    }
    if len(encoder_counts) != 1:
        raise RuntimeError("pretraining objectives use different encoders")
    encoder_initializations = {
        record["encoder_initialization_sha256"] for record in records
    }
    if len(encoder_initializations) != 1:
        raise RuntimeError("pretraining objectives use different encoder initializations")
    report = {
        "backbone": {
            "expert_kernels": [7],
            "expert_output_dim": 299,
            "gate_type": "uniform",
            "relation_mode": "none",
            "norm_type": "local_filter",
        },
        "same_encoder_parameter_budget": True,
        "same_seeded_encoder_initialization": True,
        "same_optimizer_steps_and_input_masks_required": True,
        "decoder_budget_matched": False,
        "decoder_budget_note": (
            "The waveform decoder necessarily emits 1000 samples per channel "
            "and is larger than the three statistic heads. It is treated as a "
            "stronger-capacity masked-context baseline; no decoder-budget-matched "
            "claim is made."
        ),
        "objectives": records,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not all(record["finite"] for record in records):
        raise SystemExit("an objective produced non-finite tensors")


if __name__ == "__main__":
    main()
