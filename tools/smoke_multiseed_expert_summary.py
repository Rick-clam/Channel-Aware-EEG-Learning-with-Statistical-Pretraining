"""Generate synthetic artifacts and smoke-test expert-set selection."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from summarize_tuev_expert_screen import VARIANTS, expected_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wide_dim", type=int, default=334)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = expected_configs(args.wide_dim)
    with tempfile.TemporaryDirectory() as directory:
        result_dir = Path(directory)
        sample_count = 180
        labels = np.arange(sample_count, dtype=np.int64) % 6
        subject_ids = np.asarray(
            [f"subject-{index // 10:02d}" for index in range(sample_count)]
        )
        sample_ids = np.asarray(
            [f"sample-{index:03d}" for index in range(sample_count)]
        )
        corruption_stride = {
            "single_k7": 4,
            "duplicate_k7": 5,
            "multiscale_k3_7_15": None,
            "wide_single_k7": 6,
        }
        for seed in (2026, 2027, 2028):
            for variant in VARIANTS:
                predictions = labels.copy()
                stride = corruption_stride[variant]
                if stride is not None:
                    offset = seed % stride
                    corrupt = np.arange(offset, sample_count, stride)
                    predictions[corrupt] = (predictions[corrupt] + 1) % 6
                logits = np.full((sample_count, 6), -4.0, dtype=np.float32)
                logits[np.arange(sample_count), predictions] = 4.0
                run_id = (
                    f"sealedv3_TUEV_expert_{variant}_none_uniform_seed{seed}"
                )
                prediction_path = result_dir / f"{run_id}.npz"
                np.savez_compressed(
                    prediction_path,
                    logits=logits,
                    labels=labels,
                    sample_ids=sample_ids,
                    subject_ids=subject_ids,
                )
                record = {
                    "run_id": run_id,
                    "status": "completed",
                    "metrics": {"val_cohen": 0.0},
                    "config": {
                        "dataset": "TUEV",
                        "sample_length": 5,
                        "evaluation_mode": "validation_only",
                        "gate_type": "uniform",
                        "expert_axis": "temporal",
                        "norm_type": "local_filter",
                        "relation_mode": "none",
                        "match_dynamic_budget": False,
                        "expert_kernels": configs[variant]["expert_kernels"],
                        "feature_out": configs[variant]["feature_out"],
                        "seed": seed,
                    },
                    "diagnostics": {
                        "test_set_accessed": False,
                        "test_loader_constructed": False,
                        "source_snapshot_at_start": {
                            "source_snapshot_sha256": "synthetic"
                        },
                        "validation_prediction_artifact": {
                            "path": str(prediction_path)
                        },
                    },
                }
                (result_dir / f"{run_id}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
        summary_path = result_dir / "summary.json"
        subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "summarize_tuev_expert_screen.py"),
                "--result_dir",
                str(result_dir),
                "--seeds",
                "2026",
                "2027",
                "2028",
                "--wide_dim",
                str(args.wide_dim),
                "--num_bootstrap",
                "500",
                "--output",
                str(summary_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["selected_variant_by_mean_kappa"] != "multiscale_k3_7_15":
            raise AssertionError("synthetic best expert variant was not selected")
        if not summary["predeclared_expert_gate"]["pass"]:
            raise AssertionError("synthetic expert superiority gate did not pass")
        report = {
            "selected_variant": summary["selected_variant_by_mean_kappa"],
            "expert_gate_pass": True,
            "multiplicity_control": summary["predeclared_expert_gate"][
                "multiplicity_control"
            ],
            "num_bootstrap": summary["num_bootstrap"],
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
