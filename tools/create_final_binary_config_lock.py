"""Freeze one TUAB or CHB-MIT configuration after validation-only selection."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LOCK_FIELDS = (
    "dataset",
    "model",
    "n_classes",
    "feature_in",
    "feature_out",
    "expert_number",
    "in_channels",
    "sample_length",
    "token_size",
    "hop_length",
    "pretrain_model_path",
    "transfer_mode",
    "top_k",
    "router_temperature",
    "router_aux_loss_coef",
    "gate_type",
    "gate_hidden_dim",
    "expert_axis",
    "norm_type",
    "relation_mode",
    "gate_descriptor_source",
    "match_dynamic_budget",
    "expert_kernels",
    "lr",
    "weight_decay",
    "batch_size",
    "sampling_rate",
    "seed",
    "checkpoint_metric",
    "resolved_checkpoint_metric",
    "deterministic",
    "split_manifest_dir",
    "allow_subject_overlap",
    "tuab_root",
    "chbmit_root",
    "chbmit_grouped_manifest",
    "chbmit_fold_index",
)

TRAINING_BUDGET_FIELDS = (
    "epochs",
    "lr",
    "weight_decay",
    "batch_size",
    "sampling_rate",
    "limit_train_batches",
    "limit_val_batches",
    "deterministic",
    "checkpoint_metric",
    "resolved_checkpoint_metric",
)


def metadata(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected_validation_result", type=Path, required=True
    )
    parser.add_argument(
        "--compared_validation_result",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--selection_rule",
        required=True,
        help="Predeclared validation-only rule used to select this config",
    )
    parser.add_argument("--selection_summary", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = json.loads(
        args.selected_validation_result.read_text(encoding="utf-8")
    )
    selection_summary = json.loads(
        args.selection_summary.read_text(encoding="utf-8")
    )
    if (
        selection_summary.get("selected_validation_run_id")
        != selected.get("run_id")
    ):
        raise ValueError(
            "selection summary does not select the requested validation run"
        )
    if selected["config"].get("dataset") not in {"TUAB", "CHB_MIT"}:
        raise ValueError("selected result is not a binary EEG dataset")
    selected_diagnostics = selected.get("diagnostics", {})
    selected_snapshot = selected_diagnostics.get(
        "source_snapshot_at_start", {}
    )
    selected_snapshot_hash = selected_snapshot.get("source_snapshot_sha256")
    if not selected_snapshot_hash:
        raise ValueError("selected result lacks a start-time source snapshot")

    evidence = []
    all_results = list(args.compared_validation_result)
    if args.selected_validation_result not in all_results:
        all_results.append(args.selected_validation_result)
    for path in all_results:
        record = json.loads(path.read_text(encoding="utf-8"))
        diagnostics = record.get("diagnostics", {})
        if record["config"].get("evaluation_mode") != "validation_only":
            raise ValueError(f"{path} is not validation_only")
        if diagnostics.get("test_set_accessed") is not False:
            raise ValueError(f"{path} does not prove test lock")
        if diagnostics.get("test_loader_constructed") is not False:
            raise ValueError(f"{path} constructed a test loader")
        if record["config"].get("dataset") != selected["config"].get(
            "dataset"
        ):
            raise ValueError("comparison records mix datasets")
        snapshot_hash = diagnostics.get("source_snapshot_at_start", {}).get(
            "source_snapshot_sha256"
        )
        if snapshot_hash != selected_snapshot_hash:
            raise ValueError("candidate results use different source snapshots")
        for field in TRAINING_BUDGET_FIELDS:
            if record["config"].get(field) != selected["config"].get(field):
                raise ValueError(
                    f"candidate training budget differs for {field}"
                )
        evidence.append(
            {
                **metadata(path),
                "run_id": record["run_id"],
                "metrics": record["metrics"],
            }
        )

    locked_config = {
        field: selected["config"][field] for field in LOCK_FIELDS
    }
    checkpoint = selected_diagnostics["best_checkpoint"]
    if metadata(checkpoint["path"])["sha256"] != checkpoint["sha256"]:
        raise ValueError("selected checkpoint hash no longer matches")
    prediction = selected_diagnostics["validation_prediction_artifact"]
    if metadata(prediction["path"])["sha256"] != prediction["sha256"]:
        raise ValueError("selected validation prediction hash no longer matches")
    with np.load(prediction["path"], allow_pickle=False) as artifact:
        thresholds = np.asarray(artifact["decision_threshold"], dtype=float)
    if thresholds.size == 0 or not np.isfinite(thresholds).all():
        raise ValueError("validation prediction has no finite threshold")
    if not np.allclose(thresholds, thresholds.flat[0], rtol=0.0, atol=1e-8):
        raise ValueError("validation prediction threshold is not constant")
    decision_threshold = float(thresholds.flat[0])
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("validation decision threshold is outside [0, 1]")
    pretrain_artifact = (
        metadata(selected["config"]["pretrain_model_path"])
        if selected["config"]["pretrain_model_path"]
        else None
    )

    lock = {
        "lock_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_config": locked_config,
        "selected_validation_run_id": selected["run_id"],
        "selection_rule": args.selection_rule,
        "validation_evidence": evidence,
        "selected_training_config": selected["config"],
        "selected_checkpoint": checkpoint,
        "selected_validation_prediction": prediction,
        "source_snapshot_at_training_start": selected_snapshot,
        "selection_summary": metadata(args.selection_summary),
        "split_manifest_sha256": metadata(args.split_manifest)["sha256"],
        "decision_threshold": decision_threshold,
        "decision_threshold_rule": (
            "Validation prevalence-matching threshold generated by the "
            "training runner and frozen before final testing."
        ),
        "pretrain_artifact": pretrain_artifact,
        "test_access_policy": (
            "This lock is created from validation-only artifacts. The final "
            "test runner loads the selected checkpoint without fitting and "
            "rejects configuration, source, checkpoint, or manifest mismatch."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(args.output, 0o444)
    lock_record = metadata(args.output)
    ledger_path = args.output.parent / "lock_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "created_utc": lock["created_utc"],
                    "lock": lock_record,
                    "selected_validation_run_id": selected["run_id"],
                }
            )
            + "\n"
        )
    print(json.dumps({**lock, "lock_artifact": lock_record}, indent=2))


if __name__ == "__main__":
    main()
