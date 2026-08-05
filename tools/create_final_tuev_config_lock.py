"""Freeze one TUEV configuration after validation-only selection."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


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
    "label_fraction",
    "label_subset_manifest",
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
    "split_seed",
    "checkpoint_metric",
    "resolved_checkpoint_metric",
    "deterministic",
    "tuev_root",
    "tuev_split_manifest",
    "tuev_file_manifest_summary",
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
    parser.add_argument("--selected_validation_result", type=Path, required=True)
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
    parser.add_argument("--routing_num_permutations", type=int, default=1000)
    parser.add_argument("--routing_num_bootstrap", type=int, default=10000)
    parser.add_argument("--routing_seed", type=int, default=2026)
    parser.add_argument(
        "--tuev_manifest",
        type=Path,
        default=Path("results/manifests/tuev_subject_split.json"),
    )
    parser.add_argument(
        "--mapping_summary",
        type=Path,
        default=Path(
            "results/manifests/tuev_processed_file_mapping_summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = json.loads(
        args.selected_validation_result.read_text(encoding="utf-8")
    )
    selection_summary = json.loads(
        args.selection_summary.read_text(encoding="utf-8")
    )
    selected_ids = selection_summary.get(
        "selected_validation_run_ids_by_seed", {}
    )
    summary_selected_id = selection_summary.get(
        "selected_validation_run_id",
        selected_ids.get(str(selected.get("config", {}).get("seed"))),
    )
    if summary_selected_id != selected.get("run_id"):
        raise ValueError(
            "selection summary does not select the requested validation run"
        )
    selected_start_snapshot = selected.get("diagnostics", {}).get(
        "source_snapshot_at_start", {}
    )
    selected_snapshot_hash = selected_start_snapshot.get(
        "source_snapshot_sha256"
    )
    if not selected_snapshot_hash:
        raise ValueError("selected result lacks a start-time source snapshot")
    evidence = []
    all_results = list(args.compared_validation_result)
    if args.selected_validation_result not in all_results:
        all_results.append(args.selected_validation_result)
    for path in all_results:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["config"].get("evaluation_mode") != "validation_only":
            raise ValueError(f"{path} is not validation_only")
        if record.get("diagnostics", {}).get("test_set_accessed") is not False:
            raise ValueError(f"{path} does not prove test lock")
        if record.get("diagnostics", {}).get("test_loader_constructed") is not False:
            raise ValueError(f"{path} constructed a test loader")
        current_snapshot = record.get("diagnostics", {}).get(
            "source_snapshot_at_start", {}
        ).get("source_snapshot_sha256")
        if current_snapshot != selected_snapshot_hash:
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
    checkpoint = selected["diagnostics"]["best_checkpoint"]
    if metadata(checkpoint["path"])["sha256"] != checkpoint["sha256"]:
        raise ValueError("selected checkpoint hash no longer matches")
    validation_prediction = selected["diagnostics"][
        "validation_prediction_artifact"
    ]
    if (
        metadata(validation_prediction["path"])["sha256"]
        != validation_prediction["sha256"]
    ):
        raise ValueError("selected validation prediction hash no longer matches")
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
        "selected_validation_prediction": validation_prediction,
        "source_snapshot_at_training_start": selected_start_snapshot,
        "pretrain_artifact": pretrain_artifact,
        "selection_summary": metadata(args.selection_summary),
        "tuev_manifest_sha256": metadata(args.tuev_manifest)["sha256"],
        "tuev_mapping_summary_sha256": metadata(
            args.mapping_summary
        )["sha256"],
        "routing_analysis": {
            "num_permutations": args.routing_num_permutations,
            "num_subject_bootstrap": args.routing_num_bootstrap,
            "seed": args.routing_seed,
            "global_permutation": "Sattolo full-test derangement",
            "blocked_permutation": "within-subject Sattolo derangement",
        },
        "test_access_policy": (
            "This lock is created from validation-only artifacts. The final "
            "test runner loads the selected checkpoint without fitting and "
            "rejects any configuration, source, checkpoint, or manifest "
            "mismatch."
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
