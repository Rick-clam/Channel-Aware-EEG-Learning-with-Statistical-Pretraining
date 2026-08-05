"""Summarize a multi-seed validation-only TUEV relation screen."""

import argparse
import json
from pathlib import Path

import numpy as np


RELATIONS = (
    "none",
    "attention",
    "static",
    "spatial1x1",
    "static_conditioned_matched",
    "spatial1x1_conditioned_matched",
    "dynamic",
    "dynamic_normalized",
)


def kappa(labels, predictions, classes):
    count = classes.shape[0]
    encoded = labels * count + predictions
    matrix = np.bincount(encoded, minlength=count * count).reshape(
        count, count
    )
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = (matrix.sum(0) * matrix.sum(1)).sum() / (total * total)
    if expected == 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def bootstrap_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "bootstrap_standard_error": float(values.std(ddof=1)),
        "ci_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "probability_gt_zero": float((values > 0).mean()),
    }


def holm_adjust(p_values):
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=Path, default=Path("results/validation_screen")
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028]
    )
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=32026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        raise ValueError("confirmatory relation selection requires >=3 seeds")

    records = {}
    artifacts = {}
    source_snapshots = set()
    for seed in args.seeds:
        records[str(seed)] = {}
        artifacts[seed] = {}
        for relation in RELATIONS:
            run_id = (
                "sealedv2_TUEV_temporal_local_filter_"
                f"{relation}_uniform_k3-7-15_d100_seed{seed}"
            )
            result_path = args.result_dir / f"{run_id}.json"
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            record = json.loads(result_path.read_text(encoding="utf-8"))
            config = record.get("config", {})
            diagnostics = record.get("diagnostics", {})
            expected = {
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
            }
            mismatches = {
                key: {"expected": value, "actual": config.get(key)}
                for key, value in expected.items()
                if config.get(key) != value
            }
            if mismatches:
                raise ValueError(f"{run_id} config mismatch: {mismatches}")
            if diagnostics.get("test_set_accessed") is not False:
                raise ValueError(f"{run_id} does not prove test isolation")
            if diagnostics.get("test_loader_constructed") is not False:
                raise ValueError(f"{run_id} constructed a test loader")
            snapshot = diagnostics.get("source_snapshot_at_start", {}).get(
                "source_snapshot_sha256"
            )
            if not snapshot:
                raise ValueError(f"{run_id} lacks a start source snapshot")
            source_snapshots.add(snapshot)
            prediction_record = diagnostics[
                "validation_prediction_artifact"
            ]
            prediction_path = Path(prediction_record["path"])
            with np.load(prediction_path, allow_pickle=False) as artifact:
                artifacts[seed][relation] = {
                    name: artifact[name]
                    for name in (
                        "logits",
                        "labels",
                        "sample_ids",
                        "subject_ids",
                    )
                }
            records[str(seed)][relation] = {
                "run_id": run_id,
                "result_path": str(result_path),
                "validation_prediction_artifact": str(prediction_path),
                "metrics": record["metrics"],
            }
    if len(source_snapshots) != 1:
        raise ValueError("candidate runs use different executable sources")

    point_by_seed = {}
    subject_indices = {}
    classes_by_seed = {}
    predictions = {}
    for seed in args.seeds:
        reference = artifacts[seed][RELATIONS[0]]
        for relation, artifact in artifacts[seed].items():
            for field in ("labels", "sample_ids", "subject_ids"):
                if not np.array_equal(reference[field], artifact[field]):
                    raise ValueError(
                        f"seed {seed} {relation} differs in {field}"
                    )
        labels = reference["labels"].astype(np.int64)
        classes = np.unique(labels)
        if not np.array_equal(classes, np.arange(classes.shape[0])):
            raise ValueError("TUEV labels are not contiguous from zero")
        classes_by_seed[seed] = classes
        predictions[seed] = {
            relation: artifact["logits"].argmax(axis=1).astype(np.int64)
            for relation, artifact in artifacts[seed].items()
        }
        point_by_seed[str(seed)] = {
            relation: kappa(labels, prediction, classes)
            for relation, prediction in predictions[seed].items()
        }
        subjects = np.unique(reference["subject_ids"])
        subject_indices[seed] = {
            subject: np.flatnonzero(reference["subject_ids"] == subject)
            for subject in subjects
        }

    mean_kappa = {
        relation: float(
            np.mean(
                [point_by_seed[str(seed)][relation] for seed in args.seeds]
            )
        )
        for relation in RELATIONS
    }
    std_kappa = {
        relation: float(
            np.std(
                [point_by_seed[str(seed)][relation] for seed in args.seeds],
                ddof=1,
            )
        )
        for relation in RELATIONS
    }
    selected_relation = max(mean_kappa, key=mean_kappa.get)
    controls = [
        relation for relation in RELATIONS if relation != "dynamic_normalized"
    ]

    rng = np.random.default_rng(args.bootstrap_seed)
    deltas = {
        relation: np.empty(args.num_bootstrap, dtype=np.float64)
        for relation in controls
    }
    simultaneous_min_delta = np.empty(args.num_bootstrap, dtype=np.float64)
    for iteration in range(args.num_bootstrap):
        sampled_seeds = rng.choice(
            args.seeds, size=len(args.seeds), replace=True
        )
        iteration_deltas = {relation: [] for relation in controls}
        for sampled_seed in sampled_seeds:
            seed = int(sampled_seed)
            reference = artifacts[seed][RELATIONS[0]]
            labels = reference["labels"].astype(np.int64)
            subjects = np.asarray(list(subject_indices[seed]))
            sampled_subjects = rng.choice(
                subjects, size=subjects.shape[0], replace=True
            )
            indices = np.concatenate(
                [subject_indices[seed][subject] for subject in sampled_subjects]
            )
            sampled_labels = labels[indices]
            candidate_kappa = kappa(
                sampled_labels,
                predictions[seed]["dynamic_normalized"][indices],
                classes_by_seed[seed],
            )
            for relation in controls:
                iteration_deltas[relation].append(
                    candidate_kappa
                    - kappa(
                        sampled_labels,
                        predictions[seed][relation][indices],
                        classes_by_seed[seed],
                    )
                )
        for relation in controls:
            deltas[relation][iteration] = np.mean(
                iteration_deltas[relation]
            )
        simultaneous_min_delta[iteration] = min(
            deltas[relation][iteration] for relation in controls
        )

    raw_p = {
        relation: float(
            (1 + np.count_nonzero(values <= 0.0))
            / (args.num_bootstrap + 1)
        )
        for relation, values in deltas.items()
    }
    adjusted_p = holm_adjust(raw_p)
    paired = {
        f"dynamic_normalized_minus_{relation}": {
            "point_mean_delta": float(
                mean_kappa["dynamic_normalized"] - mean_kappa[relation]
            ),
            **bootstrap_summary(deltas[relation]),
            "one_sided_bootstrap_p": raw_p[relation],
            "holm_adjusted_p": adjusted_p[relation],
        }
        for relation in controls
    }
    simultaneous_interval = bootstrap_summary(simultaneous_min_delta)
    mechanism_gate_pass = bool(
        selected_relation == "dynamic_normalized"
        and simultaneous_interval["ci_95"][0] > 0.0
    )
    selected_ids = {
        str(seed): records[str(seed)][selected_relation]["run_id"]
        for seed in args.seeds
    }
    report = {
        "protocol": (
            "validation-only, three-seed TUEV relation screen with "
            "hierarchical seed/subject bootstrap"
        ),
        "test_set_accessed": False,
        "seeds": args.seeds,
        "num_bootstrap": args.num_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "relations": list(RELATIONS),
        "records": records,
        "source_snapshot_sha256": next(iter(source_snapshots)),
        "point_validation_cohen_kappa_by_seed": point_by_seed,
        "mean_validation_cohen_kappa": mean_kappa,
        "std_validation_cohen_kappa": std_kappa,
        "selected_relation_by_mean_kappa": selected_relation,
        "selected_validation_run_ids_by_seed": selected_ids,
        "paired_hierarchical_bootstrap_kappa_deltas": paired,
        "simultaneous_minimum_delta_vs_all_controls": simultaneous_interval,
        "predeclared_mechanism_gate": {
            "rule": (
                "dynamic_normalized has the largest mean validation Cohen "
                "kappa across >=3 seeds and the hierarchical-bootstrap 95% "
                "lower bound of its minimum delta over all seven controls "
                "is strictly above zero"
            ),
            "multiplicity_control": (
                "simultaneous minimum-delta confidence bound; per-control "
                "one-sided bootstrap p-values additionally use Holm adjustment"
            ),
            "pass": mechanism_gate_pass,
            "next_action": (
                "continue gate/expert ablations"
                if mechanism_gate_pass
                else "drop or narrow the dynamic-relation superiority claim"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
