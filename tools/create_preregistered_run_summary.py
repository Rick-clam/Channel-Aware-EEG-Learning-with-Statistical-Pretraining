"""Record that one Top-K configuration was fixed before final test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.validation_result.read_text(encoding="utf-8"))
    config = result.get("config", {})
    diagnostics = result.get("diagnostics", {})
    if config.get("evaluation_mode") != "validation_only":
        raise ValueError("input result is not validation_only")
    if diagnostics.get("test_set_accessed") is not False:
        raise ValueError("input result does not prove the test lock")
    if diagnostics.get("test_loader_constructed") is not False:
        raise ValueError("input result constructed a test loader")
    run_id = result["run_id"]
    summary = {
        "selection_type": "preregistered_configuration_lock",
        "selected_validation_run_id": run_id,
        "selected_validation_run_ids_by_seed": {
            str(config["seed"]): run_id
        },
        "dataset": config["dataset"],
        "seed": config["seed"],
        "top_k": config["top_k"],
        "rule": (
            "K, seed, architecture, and training budget were preregistered; "
            "validation selected only the checkpoint/epoch and binary threshold."
        ),
        "validation_metrics": result["metrics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
