"""Exercise TUEV evaluation-lock failure and success paths."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_multi_moe import FINAL_LOCK_FIELDS, verify_evaluation_lock


SOURCE = {"source_snapshot_sha256": "synthetic-source-snapshot"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_args(root):
    values = {
        "dataset": "TUEV",
        "model": "dwmoespace_newgate",
        "n_classes": 6,
        "feature_in": 16000,
        "feature_out": 100,
        "expert_number": 4,
        "in_channels": 12,
        "sample_length": 10.0,
        "token_size": 200,
        "hop_length": 100,
        "pretrain_model_path": "",
        "transfer_mode": "full_finetune",
        "label_fraction": 1.0,
        "label_subset_manifest": "results/manifests/tuev_label_subsets_seed2026.json",
        "top_k": 3,
        "gate_type": "uniform",
        "gate_hidden_dim": 16,
        "expert_axis": "temporal",
        "norm_type": "local_filter",
        "relation_mode": "dynamic_normalized",
        "gate_descriptor_source": "raw",
        "match_dynamic_budget": False,
        "expert_kernels": "3,7,15",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "sampling_rate": 200,
        "seed": 2026,
        "split_seed": 2026,
        "checkpoint_metric": "auto",
        "resolved_checkpoint_metric": "val_cohen",
        "deterministic": True,
        "tuev_root": "/synthetic/tuev",
        "tuev_split_manifest": str(root / "tuev_split.json"),
        "tuev_file_manifest_summary": str(root / "mapping_summary.json"),
        "evaluation_mode": "validation_only",
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "limit_test_batches": 1.0,
        "result_dir": str(root / "results" / "validation_screen"),
        "final_config_lock": "",
    }
    return SimpleNamespace(**values)


def expect_failure(args, fragment, snapshot=SOURCE):
    try:
        verify_evaluation_lock(args, snapshot)
    except (ValueError, FileNotFoundError) as error:
        if fragment not in str(error):
            raise
        return str(error)
    raise AssertionError(f"Expected failure containing {fragment!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = base_args(root)
        split_path = Path(args.tuev_split_manifest)
        mapping_path = Path(args.tuev_file_manifest_summary)
        split_path.write_text('{"dataset":"TUEV"}\n', encoding="utf-8")
        mapping_path.write_text('{"mapping":"synthetic"}\n', encoding="utf-8")
        checkpoint_path = root / "selected.ckpt"
        checkpoint_path.write_bytes(b"synthetic-checkpoint")
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        }
        report = {
            "validation_only": verify_evaluation_lock(args, SOURCE) is None
        }

        args.evaluation_mode = "smoke_test"
        args.result_dir = str(root / "results" / "smoke")
        report["unlimited_smoke_rejected"] = expect_failure(
            args, "at least one limited-batch"
        )
        args.limit_train_batches = 0.1
        report["limited_smoke"] = verify_evaluation_lock(args, SOURCE)

        args.evaluation_mode = "final_test"
        args.limit_train_batches = 1.0
        report["missing_lock_rejected"] = expect_failure(
            args, "requires --final_config_lock"
        )
        locked_config = {
            field: getattr(args, field) for field in FINAL_LOCK_FIELDS
        }
        lock = {
            "lock_version": 2,
            "locked_config": locked_config,
            "validation_evidence": [{"run_id": "validation-only-smoke"}],
            "tuev_manifest_sha256": digest(split_path),
            "tuev_mapping_summary_sha256": digest(mapping_path),
            "source_snapshot_at_training_start": SOURCE,
            "selected_checkpoint": checkpoint,
            "pretrain_artifact": None,
            "routing_analysis": {
                "num_permutations": 1000,
                "num_subject_bootstrap": 10000,
                "seed": 2026,
            },
        }
        lock_path = root / "lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        os.chmod(lock_path, 0o444)
        lock_record = {
            "path": str(lock_path),
            "sha256": digest(lock_path),
            "size_bytes": lock_path.stat().st_size,
        }
        (root / "lock_ledger.jsonl").write_text(
            json.dumps({"lock": lock_record}) + "\n", encoding="utf-8"
        )
        args.final_config_lock = str(lock_path)
        report["valid_final_lock"] = verify_evaluation_lock(args, SOURCE)
        args.seed = 2027
        report["config_mismatch_rejected"] = expect_failure(
            args, "differs from lock"
        )
        args.seed = 2026
        report["source_mismatch_rejected"] = expect_failure(
            args,
            "source differs",
            {"source_snapshot_sha256": "different"},
        )

    rendered = json.dumps(report, indent=2)
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
