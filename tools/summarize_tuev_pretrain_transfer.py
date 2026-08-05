"""Summarize the registered TUEV pretraining transfer gate."""

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "scratch",
    "unmasked_stats",
    "normalized_masked_stats",
    "shuffled_masked_stats",
    "masked_waveform",
)
CANDIDATE = "normalized_masked_stats"


def kappa(labels, predictions):
    classes = int(max(labels.max(), predictions.max())) + 1
    matrix = np.bincount(
        labels * classes + predictions, minlength=classes * classes
    ).reshape(classes, classes)
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = (matrix.sum(0) * matrix.sum(1)).sum() / (total * total)
    return 0.0 if expected == 1.0 else float((observed - expected) / (1 - expected))


def bootstrap_summary(values):
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
        "--result_dir", type=Path, default=Path("results/pretrain_transfer_tuev")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument(
        "--transfer_mode",
        choices=("frozen_linear", "full_finetune"),
        default="frozen_linear",
    )
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--allow_screening_single_seed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) < 3 and not args.allow_screening_single_seed:
        raise ValueError("confirmatory transfer inference requires at least three seeds")

    fraction_tag = f"{args.label_fraction:g}".replace(".", "p")
    artifacts = {}
    records = {}
    snapshots = set()
    for seed in args.seeds:
        artifacts[seed] = {}
        records[str(seed)] = {}
        for variant in VARIANTS:
            run_id = (
                f"tuev_{variant}_{args.transfer_mode}_"
                f"labels{fraction_tag}_seed{seed}"
            )
            result_path = args.result_dir / f"{run_id}.json"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            config = record["config"]
            expected = {
                "dataset": "TUEV",
                "model": "dwmoespace_newgate",
                "feature_out": 299,
                "expert_axis": "temporal",
                "expert_kernels": "7",
                "gate_type": "uniform",
                "relation_mode": "none",
                "norm_type": "local_filter",
                "transfer_mode": args.transfer_mode,
                "label_fraction": args.label_fraction,
                "seed": seed,
                "evaluation_mode": "validation_only",
            }
            mismatch = {
                key: (value, config.get(key))
                for key, value in expected.items()
                if config.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"{run_id} config mismatch: {mismatch}")
            if variant == "scratch" and config.get("pretrain_model_path"):
                raise RuntimeError("scratch result loads a pretrained checkpoint")
            if variant != "scratch" and not config.get("pretrain_model_path"):
                raise RuntimeError(f"{variant} lacks a pretrained checkpoint")
            diagnostics = record["diagnostics"]
            if diagnostics.get("test_loader_constructed") is not False:
                raise RuntimeError(f"{run_id} constructed a test loader")
            if diagnostics.get("test_set_accessed") is not False:
                raise RuntimeError(f"{run_id} accessed the test set")
            snapshots.add(
                diagnostics["source_snapshot_at_start"][
                    "source_snapshot_sha256"
                ]
            )
            path = Path(
                diagnostics["validation_prediction_artifact"]["path"]
            )
            with np.load(path, allow_pickle=False) as data:
                artifacts[seed][variant] = {
                    key: data[key]
                    for key in ("logits", "labels", "sample_ids", "subject_ids")
                }
            records[str(seed)][variant] = {
                "result": str(result_path),
                "predictions": str(path),
                "metrics": record["metrics"],
            }
    if len(snapshots) != 1:
        raise RuntimeError("TUEV transfer runs use different executable sources")

    points = {}
    predictions = {}
    subject_indices = {}
    for seed in args.seeds:
        reference = artifacts[seed][VARIANTS[0]]
        if reference["labels"].shape != (6744,):
            raise RuntimeError("unexpected TUEV validation window count")
        for variant in VARIANTS[1:]:
            for field in ("labels", "sample_ids", "subject_ids"):
                if not np.array_equal(reference[field], artifacts[seed][variant][field]):
                    raise RuntimeError(f"seed {seed} {variant} differs in {field}")
        labels = reference["labels"].astype(np.int64)
        predictions[seed] = {
            variant: artifacts[seed][variant]["logits"].argmax(1).astype(np.int64)
            for variant in VARIANTS
        }
        points[str(seed)] = {
            variant: kappa(labels, predictions[seed][variant])
            for variant in VARIANTS
        }
        subjects = np.unique(reference["subject_ids"])
        if subjects.shape[0] != 29:
            raise RuntimeError("unexpected TUEV validation subject count")
        subject_indices[seed] = {
            subject: np.flatnonzero(reference["subject_ids"] == subject)
            for subject in subjects
        }

    means = {
        variant: float(np.mean([points[str(seed)][variant] for seed in args.seeds]))
        for variant in VARIANTS
    }
    standard_deviations = {
        variant: (
            float(np.std([points[str(seed)][variant] for seed in args.seeds], ddof=1))
            if len(args.seeds) > 1
            else None
        )
        for variant in VARIANTS
    }
    controls = [variant for variant in VARIANTS if variant != CANDIDATE]
    rng = np.random.default_rng(args.bootstrap_seed)
    deltas = {
        control: np.empty(args.num_bootstrap, dtype=np.float64)
        for control in controls
    }
    simultaneous = np.empty(args.num_bootstrap, dtype=np.float64)
    for iteration in range(args.num_bootstrap):
        sampled_seeds = rng.choice(args.seeds, size=len(args.seeds), replace=True)
        values = {control: [] for control in controls}
        for sampled_seed in sampled_seeds:
            seed = int(sampled_seed)
            reference = artifacts[seed][VARIANTS[0]]
            labels = reference["labels"].astype(np.int64)
            subjects = np.asarray(list(subject_indices[seed]))
            sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
            indices = np.concatenate(
                [subject_indices[seed][subject] for subject in sampled_subjects]
            )
            candidate_kappa = kappa(labels[indices], predictions[seed][CANDIDATE][indices])
            for control in controls:
                values[control].append(
                    candidate_kappa
                    - kappa(labels[indices], predictions[seed][control][indices])
                )
        for control in controls:
            deltas[control][iteration] = np.mean(values[control])
        simultaneous[iteration] = min(deltas[name][iteration] for name in controls)

    raw_two_sided_p = {
        control: float(
            min(
                1.0,
                2
                * min(
                    (1 + np.count_nonzero(values <= 0)) / (args.num_bootstrap + 1),
                    (1 + np.count_nonzero(values >= 0)) / (args.num_bootstrap + 1),
                ),
            )
        )
        for control, values in deltas.items()
    }
    adjusted_p = holm_adjust(raw_two_sided_p)
    paired = {
        f"{CANDIDATE}_minus_{control}": {
            "point_mean_delta": means[CANDIDATE] - means[control],
            **bootstrap_summary(deltas[control]),
            "two_sided_bootstrap_p": raw_two_sided_p[control],
            "holm_adjusted_p": adjusted_p[control],
        }
        for control in controls
    }
    simultaneous_summary = bootstrap_summary(simultaneous)
    confirmatory = len(set(args.seeds)) >= 3
    gate_pass = bool(
        confirmatory
        and max(means, key=means.get) == CANDIDATE
        and simultaneous_summary["ci_95"][0] > 0
    )
    report = {
        "protocol": "TUEV held-out frozen-transfer pretraining objective gate",
        "confirmatory": confirmatory,
        "test_set_accessed": False,
        "transfer_mode": args.transfer_mode,
        "label_fraction": args.label_fraction,
        "seeds": args.seeds,
        "num_bootstrap": args.num_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "records": records,
        "source_snapshot_sha256": next(iter(snapshots)),
        "point_validation_cohen_kappa_by_seed": points,
        "mean_validation_cohen_kappa": means,
        "std_validation_cohen_kappa": standard_deviations,
        "selected_variant_by_mean_kappa": max(means, key=means.get),
        "paired_hierarchical_bootstrap_kappa_deltas": paired,
        "simultaneous_minimum_delta_vs_all_controls": simultaneous_summary,
        "predeclared_transfer_gate": {
            "pass": gate_pass,
            "rule": (
                "normalized_masked_stats has the largest mean kappa across >=3 "
                "seeds and the 95% lower bound of its simultaneous minimum "
                "delta over all four controls is greater than zero"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
