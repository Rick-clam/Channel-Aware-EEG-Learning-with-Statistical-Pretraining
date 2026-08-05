"""Synthetic forward/backward audit for every registered pretraining objective."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_unsupervised_pretrain import LitModel_supervised_pretrain


OBJECTIVES = (
    "unmasked_stats",
    "normalized_masked_stats",
    "shuffled_masked_stats",
    "masked_waveform",
)


def make_args(objective, normalization_path):
    return SimpleNamespace(
        feature_out=299,
        need_token=20,
        sampling_rate=200,
        window_seconds=5,
        gate_type="uniform",
        gate_hidden_dim=16,
        expert_axis="temporal",
        relation_mode="none",
        norm_type="local_filter",
        gate_descriptor_source="raw",
        match_dynamic_budget=False,
        expert_kernels="7",
        objective=objective,
        stat_normalization=str(normalization_path),
        logdataset="tuabchbmit",
        mask_ratio=0.5,
        lr=1e-4,
        weight_decay=1e-5,
        seed=2026,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    device = (
        torch.device("cuda")
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )
    torch.manual_seed(2026)
    batch = torch.randn(2, 16, 1000, device=device)
    reports = []
    with tempfile.TemporaryDirectory() as directory:
        normalization_path = Path(directory) / "normalization.json"
        normalization_path.write_text(
            json.dumps(
                {
                    "logdataset": "tuabchbmit",
                    "sampling_rate_hz": 200,
                    "window_seconds": 5,
                    "window_samples": 1000,
                    "segment_count": 20,
                    "stats_features": ["mean", "std", "skewness"],
                    "mean": [0.1, 0.8, -0.05],
                    "std": [0.4, 0.3, 0.9],
                }
            ),
            encoding="utf-8",
        )
        for objective in OBJECTIVES:
            objective_args = make_args(
                objective,
                normalization_path
                if "masked_stats" in objective
                else "",
            )
            lightning_model = LitModel_supervised_pretrain(
                objective_args, directory
            ).to(device)
            # This audit calls the Lightning step without attaching a Trainer.
            lightning_model.log = lambda *_args, **_kwargs: None
            torch.manual_seed(2026)
            loss = lightning_model.training_step(batch, 0)
            loss.backward()
            missing = [
                name
                for name, parameter in lightning_model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            mask = lightning_model._segment_mask(batch.shape[0], batch.device)
            masked_input = lightning_model._masked_encoder_input(batch, mask)
            segmented_original = lightning_model.model.segment_input(batch)
            segmented_masked = lightning_model.model.segment_input(masked_input)
            masked_values_zero = bool(
                torch.count_nonzero(
                    segmented_masked[mask[:, None, :, None].expand_as(segmented_masked)]
                ).item()
                == 0
            )
            visible_values_equal = bool(
                torch.equal(
                    segmented_masked[
                        (~mask)[:, None, :, None].expand_as(segmented_masked)
                    ],
                    segmented_original[
                        (~mask)[:, None, :, None].expand_as(segmented_original)
                    ],
                )
            )
            target, prediction = lightning_model.model(
                batch,
                masked_input if objective != "unmasked_stats" else batch,
            )
            reports.append(
                {
                    "objective": objective,
                    "device": str(device),
                    "loss": float(loss.detach().cpu()),
                    "finite_loss": bool(torch.isfinite(loss)),
                    "target_shape": list(target.shape),
                    "prediction_shape": list(prediction.shape),
                    "mask_count_per_sample": mask.sum(dim=1).cpu().tolist(),
                    "mask_indices": [
                        torch.nonzero(row, as_tuple=False).flatten().cpu().tolist()
                        for row in mask
                    ],
                    "masked_values_zero": masked_values_zero,
                    "visible_values_equal": visible_values_equal,
                    "missing_gradients": missing,
                }
            )

    report = {
        "backbone": {
            "expert_axis": "temporal",
            "expert_kernels": [7],
            "expert_output_dim": 299,
            "relation_mode": "none",
            "gate_type": "uniform",
            "norm_type": "local_filter",
        },
        "input_shape": list(batch.shape),
        "objectives": reports,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if any(not record["finite_loss"] for record in reports):
        raise SystemExit("a pretraining objective produced non-finite loss")
    if any(record["missing_gradients"] for record in reports):
        raise SystemExit("a pretraining objective has missing gradients")
    if any(
        not record["masked_values_zero"] or not record["visible_values_equal"]
        for record in reports
    ):
        raise SystemExit("segment masking invariant failed")
    masked_records = [
        record for record in reports if record["objective"] != "unmasked_stats"
    ]
    if any(
        record["mask_indices"] != masked_records[0]["mask_indices"]
        for record in masked_records[1:]
    ):
        raise SystemExit("masked objectives do not share the exact mask schedule")


if __name__ == "__main__":
    main()
