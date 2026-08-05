"""Generate synthetic artifacts and smoke-test multi-seed relation selection."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from summarize_tuev_relation_screen import RELATIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        result_dir = Path(directory)
        sample_count = 72
        labels = np.arange(sample_count, dtype=np.int64) % 6
        subject_ids = np.asarray(
            [f"subject-{index // 6:02d}" for index in range(sample_count)]
        )
        sample_ids = np.asarray([f"sample-{index:03d}" for index in range(sample_count)])
        for seed in (2026, 2027, 2028):
            for relation_index, relation in enumerate(RELATIONS):
                predictions = labels.copy()
                if relation != "dynamic_normalized":
                    stride = 6 - min(relation_index, 5)
                    corrupt = np.arange(relation_index % 3, sample_count, stride)
                    predictions[corrupt] = (predictions[corrupt] + 1) % 6
                logits = np.full((sample_count, 6), -4.0, dtype=np.float32)
                logits[np.arange(sample_count), predictions] = 4.0
                run_id = (
                    "sealedv2_TUEV_temporal_local_filter_"
                    f"{relation}_uniform_k3-7-15_d100_seed{seed}"
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
                    "metrics": {"cohen_kappa": 0.0},
                    "config": {
                        "dataset": "TUEV",
                        "sample_length": 5,
                        "evaluation_mode": "validation_only",
                        "gate_type": "uniform",
                        "expert_axis": "temporal",
                        "norm_type": "local_filter",
                        "relation_mode": relation,
                        "match_dynamic_budget": True,
                        "expert_kernels": "3,7,15",
                        "feature_out": 100,
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
                str(root / "tools" / "summarize_tuev_relation_screen.py"),
                "--result_dir",
                str(result_dir),
                "--seeds",
                "2026",
                "2027",
                "2028",
                "--num_bootstrap",
                "500",
                "--output",
                str(summary_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["selected_relation_by_mean_kappa"] != "dynamic_normalized":
            raise AssertionError("synthetic best relation was not selected")
        if not summary["predeclared_mechanism_gate"]["pass"]:
            raise AssertionError("synthetic superiority gate did not pass")
        report = {
            "selected_relation": summary[
                "selected_relation_by_mean_kappa"
            ],
            "selected_validation_run_ids_by_seed": summary[
                "selected_validation_run_ids_by_seed"
            ],
            "mechanism_gate_pass": True,
            "multiplicity_control": summary["predeclared_mechanism_gate"][
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
