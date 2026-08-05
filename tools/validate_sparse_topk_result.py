"""Audit one sparse Top-K validation or locked-test result artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact(record: dict, key: str) -> None:
    artifact = record["diagnostics"][key]
    path = Path(artifact["path"])
    if not path.is_file() or sha256(path) != artifact["sha256"]:
        raise ValueError(f"{key} is missing or has a hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    record = json.loads(args.result.read_text(encoding="utf-8"))
    config = record["config"]
    diagnostics = record["diagnostics"]
    if record.get("status") != "completed":
        raise ValueError("result is not completed")
    if config.get("model") != "dwmoespace_sparse_topk":
        raise ValueError("result is not the sparse Top-K model")
    top_k = int(config["top_k"])
    if top_k not in (1, 2, 3):
        raise ValueError("Top-K is outside the frozen range")
    if config.get("expert_kernels") != "3,7,15":
        raise ValueError("unexpected expert kernels")
    if float(config.get("router_aux_loss_coef")) != 0.01:
        raise ValueError("unexpected router auxiliary coefficient")
    if float(config.get("router_temperature")) != 1.0:
        raise ValueError("unexpected router temperature")
    routing = diagnostics["sparse_routing"]
    if routing["top_k"] != top_k:
        raise ValueError("routing Top-K differs from configuration")
    if routing["executed_sample_expert_assignments"] != routing[
        "expected_sample_expert_assignments"
    ]:
        raise ValueError("executed sparse assignments differ from expectation")
    if routing.get("only_selected_samples_executed") is not True:
        raise ValueError("result does not prove selected-only execution")
    if abs(routing["active_expert_fraction"] - top_k / 3) > 1e-12:
        raise ValueError("active expert fraction is inconsistent")
    snapshot = diagnostics.get("source_snapshot_at_start", {})
    if not snapshot.get("source_snapshot_sha256"):
        raise ValueError("missing executable source snapshot")

    mode = config["evaluation_mode"]
    if mode == "validation_only":
        if diagnostics.get("test_set_accessed") is not False:
            raise ValueError("validation result accessed the test set")
        if diagnostics.get("test_loader_constructed") is not False:
            raise ValueError("validation result constructed a test loader")
        validate_artifact(record, "validation_prediction_artifact")
    elif mode == "final_test":
        if diagnostics.get("test_set_accessed") is not True:
            raise ValueError("final result did not access the test set")
        if diagnostics.get("test_loader_constructed") is not True:
            raise ValueError("final result did not construct a test loader")
        if diagnostics.get("evaluation_lock", {}).get("mode") != "final_test":
            raise ValueError("final result lacks a verified evaluation lock")
        validate_artifact(record, "prediction_artifact")
    elif mode != "smoke_test":
        raise ValueError(f"unsupported evaluation mode: {mode}")
    print(json.dumps({
        "status": "passed",
        "result": str(args.result),
        "dataset": config["dataset"],
        "mode": mode,
        "top_k": top_k,
        "seed": config["seed"],
        "metrics": record["metrics"],
        "active_expert_fraction": routing["active_expert_fraction"],
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
