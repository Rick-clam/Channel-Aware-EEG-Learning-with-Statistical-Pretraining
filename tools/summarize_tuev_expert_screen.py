"""Summarize the sealed multi-seed TUEV temporal-expert isolation screen."""

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "single_k7",
    "duplicate_k7",
    "multiscale_k3_7_15",
    "wide_single_k7",
)
CANDIDATE = "multiscale_k3_7_15"


def kappa(labels, predictions, classes):
    count = classes.shape[0]
    encoded = labels * count + predictions
    matrix = np.bincount(encoded, minlength=count * count).reshape(count, count)
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


def expected_configs(wide_dim):
    return {
        "single_k7": {"expert_kernels": "7", "feature_out": 100},
        "duplicate_k7": {"expert_kernels": "7,7,7", "feature_out": 100},
        "multiscale_k3_7_15": {
            "expert_kernels": "3,7,15",
            "feature_out": 100,
        },
        "wide_single_k7": {
            "expert_kernels": "7",
            "feature_out": wide_dim,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=Path, default=Path("results/expert_screen")
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028]
    )
    parser.add_argument("--wide_dim", type=int, required=True)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=42026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        raise ValueError("confirmatory expert selection requires >=3 seeds")

    variant_configs = expected_configs(args.wide_dim)
    records = {}
    artifacts = {}
    source_snapshots = set()
    for seed in args.seeds:
        records[str(seed)] = {}
        artifacts[seed] = {}
        for variant in VARIANTS:
            run_id = f"sealedv3_TUEV_expert_{variant}_none_uniform_seed{seed}"
            result_path = args.result_dir / f"{run_id}.json"
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if record.get("status") != "completed":
                raise ValueError(f"{run_id} is not completed")
            config = record.get("config", {})
            diagnostics = record.get("diagnostics", {})
            expected = {
                "dataset": "TUEV",
                "sample_length": 5,
                "evaluation_mode": "validation_only",
                "gate_type": "uniform",
                "expert_axis": "temporal",
                "norm_type": "local_filter",
                "relation_mode": "none",
                "match_dynamic_budget": False,
                "seed": seed,
                **variant_configs[variant],
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
            prediction_record = diagnostics.get("validation_prediction_artifact")
            if not prediction_record or not prediction_record.get("path"):
                raise ValueError(f"{run_id} lacks validation predictions")
            prediction_path = Path(prediction_record["path"])
            with np.load(prediction_path, allow_pickle=False) as artifact:
                artifacts[seed][variant] = {
                    name: artifact[name]
                    for name in ("logits", "labels", "sample_ids", "subject_ids")
                }
            records[str(seed)][variant] = {
                "run_id": run_id,
                "result_path": str(result_path),
                "validation_prediction_artifact": str(prediction_path),
                "metrics": record["metrics"],
            }
    if len(source_snapshots) != 1:
        raise ValueError("expert runs use different executable sources")

    point_by_seed = {}
    subject_indices = {}
    classes_by_seed = {}
    predictions = {}
    for seed in args.seeds:
        reference = artifacts[seed][VARIANTS[0]]
        for variant, artifact in artifacts[seed].items():
            for field in ("labels", "sample_ids", "subject_ids"):
                if not np.array_equal(reference[field], artifact[field]):
                    raise ValueError(f"seed {seed} {variant} differs in {field}")
        labels = reference["labels"].astype(np.int64)
        classes = np.unique(labels)
        if not np.array_equal(classes, np.arange(classes.shape[0])):
            raise ValueError("TUEV labels are not contiguous from zero")
        classes_by_seed[seed] = classes
        predictions[seed] = {
            variant: artifact["logits"].argmax(axis=1).astype(np.int64)
            for variant, artifact in artifacts[seed].items()
        }
        point_by_seed[str(seed)] = {
            variant: kappa(labels, prediction, classes)
            for variant, prediction in predictions[seed].items()
        }
        subjects = np.unique(reference["subject_ids"])
        subject_indices[seed] = {
            subject: np.flatnonzero(reference["subject_ids"] == subject)
            for subject in subjects
        }

    mean_kappa = {
        variant: float(
            np.mean([point_by_seed[str(seed)][variant] for seed in args.seeds])
        )
        for variant in VARIANTS
    }
    std_kappa = {
        variant: float(
            np.std(
                [point_by_seed[str(seed)][variant] for seed in args.seeds], ddof=1
            )
        )
        for variant in VARIANTS
    }
    selected_variant = max(VARIANTS, key=mean_kappa.get)
    controls = [variant for variant in VARIANTS if variant != CANDIDATE]

    rng = np.random.default_rng(args.bootstrap_seed)
    deltas = {
        control: np.empty(args.num_bootstrap, dtype=np.float64)
        for control in controls
    }
    simultaneous_min_delta = np.empty(args.num_bootstrap, dtype=np.float64)
    for iteration in range(args.num_bootstrap):
        sampled_seeds = rng.choice(args.seeds, size=len(args.seeds), replace=True)
        iteration_deltas = {control: [] for control in controls}
        for sampled_seed in sampled_seeds:
            seed = int(sampled_seed)
            reference = artifacts[seed][VARIANTS[0]]
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
                predictions[seed][CANDIDATE][indices],
                classes_by_seed[seed],
            )
            for control in controls:
                iteration_deltas[control].append(
                    candidate_kappa
                    - kappa(
                        sampled_labels,
                        predictions[seed][control][indices],
                        classes_by_seed[seed],
                    )
                )
        for control in controls:
            deltas[control][iteration] = np.mean(iteration_deltas[control])
        simultaneous_min_delta[iteration] = min(
            deltas[control][iteration] for control in controls
        )

    raw_p = {
        control: float(
            (1 + np.count_nonzero(values <= 0.0)) / (args.num_bootstrap + 1)
        )
        for control, values in deltas.items()
    }
    adjusted_p = holm_adjust(raw_p)
    paired = {
        f"{CANDIDATE}_minus_{control}": {
            "point_mean_delta": float(mean_kappa[CANDIDATE] - mean_kappa[control]),
            **bootstrap_summary(deltas[control]),
            "one_sided_bootstrap_p": raw_p[control],
            "holm_adjusted_p": adjusted_p[control],
        }
        for control in controls
    }
    simultaneous_interval = bootstrap_summary(simultaneous_min_delta)
    expert_gate_pass = bool(
        selected_variant == CANDIDATE
        and simultaneous_interval["ci_95"][0] > 0.0
    )
    report = {
        "protocol": (
            "sealed validation-only, three-seed TUEV temporal-expert isolation "
            "with hierarchical seed/subject bootstrap"
        ),
        "test_set_accessed": False,
        "seeds": args.seeds,
        "wide_dim": args.wide_dim,
        "num_bootstrap": args.num_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "variants": list(VARIANTS),
        "variant_configs": variant_configs,
        "records": records,
        "source_snapshot_sha256": next(iter(source_snapshots)),
        "point_validation_cohen_kappa_by_seed": point_by_seed,
        "mean_validation_cohen_kappa": mean_kappa,
        "std_validation_cohen_kappa": std_kappa,
        "selected_variant_by_mean_kappa": selected_variant,
        "paired_hierarchical_bootstrap_kappa_deltas": paired,
        "simultaneous_minimum_delta_vs_all_controls": simultaneous_interval,
        "predeclared_expert_gate": {
            "rule": (
                "multiscale_k3_7_15 has the largest mean validation Cohen kappa "
                "across >=3 seeds and the hierarchical-bootstrap 95% lower bound "
                "of its minimum delta over single-k7, duplicate-k7, and the "
                "parameter-matched wide-single control is strictly above zero"
            ),
            "multiplicity_control": (
                "simultaneous minimum-delta confidence bound; per-control "
                "one-sided bootstrap p-values additionally use Holm adjustment"
            ),
            "pass": expert_gate_pass,
            "next_action": (
                "confirm selected multiscale architecture on TUAB"
                if expert_gate_pass
                else "remove the multi-scale complementarity claim and simplify"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
