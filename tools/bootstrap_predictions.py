"""Subject-level bootstrap intervals for one or two prediction artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def load_artifact(path):
    artifact = np.load(path, allow_pickle=False)
    required = {"logits", "labels", "sample_ids", "subject_ids"}
    missing = required - set(artifact.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    output = {key: artifact[key] for key in required}
    if "decision_threshold" in artifact.files:
        thresholds = np.asarray(artifact["decision_threshold"]).reshape(-1)
        if thresholds.shape[0] != output["labels"].shape[0]:
            raise ValueError("decision_threshold has the wrong length")
        if not np.allclose(thresholds, thresholds[0]):
            raise ValueError("decision_threshold is not constant")
        output["decision_threshold"] = float(thresholds[0])
    return output


def metrics(labels, logits, classes=None, threshold=None):
    logits = np.asarray(logits)
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        probabilities = logits.reshape(-1)
        threshold = 0.5 if threshold is None else float(threshold)
        predictions = (probabilities >= threshold).astype(np.int64)
        return {
            "balanced_accuracy": balanced_accuracy_score(
                labels, predictions
            ),
            "cohen_kappa": cohen_kappa_score(labels, predictions),
            "weighted_f1": f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "accuracy": accuracy_score(labels, predictions),
            "pr_auc": average_precision_score(labels, probabilities),
            "roc_auc": roc_auc_score(labels, probabilities),
            "decision_threshold": threshold,
        }
    predictions = np.asarray(logits).argmax(axis=1)
    classes = np.unique(labels) if classes is None else np.asarray(classes)
    matrix = confusion_matrix(labels, predictions, labels=classes)
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("resample omits at least one reference class")
    balanced_accuracy = float(
        np.mean(np.diag(matrix) / support)
    )
    return {
        "balanced_accuracy": balanced_accuracy,
        "cohen_kappa": cohen_kappa_score(
            labels, predictions, labels=classes
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            labels=classes,
            average="weighted",
            zero_division=0,
        ),
        "accuracy": accuracy_score(labels, predictions),
    }


def validate_pair(first, second):
    if not np.array_equal(first["sample_ids"], second["sample_ids"]):
        raise ValueError("paired artifacts do not have identical sample order")
    if not np.array_equal(first["subject_ids"], second["subject_ids"]):
        raise ValueError("paired artifacts do not have identical subject IDs")
    if not np.array_equal(first["labels"], second["labels"]):
        raise ValueError("paired artifacts do not have identical labels")


def summarize(samples):
    array = np.asarray(samples, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_error": float(array.std(ddof=1)),
        "ci_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_a", type=Path)
    parser.add_argument("--artifact_b", type=Path)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    first = load_artifact(args.artifact_a)
    second = load_artifact(args.artifact_b) if args.artifact_b else None
    if second is not None:
        validate_pair(first, second)

    subjects = np.unique(first["subject_ids"])
    classes = np.unique(first["labels"])
    subject_indices = {
        subject: np.flatnonzero(first["subject_ids"] == subject)
        for subject in subjects
    }
    rng = np.random.default_rng(args.seed)
    first_samples = {
        metric: []
        for metric in metrics(
            first["labels"],
            first["logits"],
            classes,
            first.get("decision_threshold"),
        )
        if metric != "decision_threshold"
    }
    second_samples = (
        {metric: [] for metric in first_samples} if second is not None else None
    )
    delta_samples = (
        {metric: [] for metric in first_samples} if second is not None else None
    )

    attempts = 0
    max_attempts = max(args.num_bootstrap * 20, args.num_bootstrap + 100)
    while len(next(iter(first_samples.values()))) < args.num_bootstrap:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Too many bootstrap resamples omitted a reference class"
            )
        sampled_subjects = rng.choice(
            subjects, size=subjects.shape[0], replace=True
        )
        indices = np.concatenate(
            [subject_indices[subject] for subject in sampled_subjects]
        )
        if not np.array_equal(
            np.unique(first["labels"][indices]), classes
        ):
            continue
        metrics_a = metrics(
            first["labels"][indices],
            first["logits"][indices],
            classes,
            first.get("decision_threshold"),
        )
        for name, value in metrics_a.items():
            if name in first_samples:
                first_samples[name].append(value)
        if second is not None:
            metrics_b = metrics(
                second["labels"][indices],
                second["logits"][indices],
                classes,
                second.get("decision_threshold"),
            )
            for name, value in metrics_b.items():
                if name in second_samples:
                    second_samples[name].append(value)
                    delta_samples[name].append(value - metrics_a[name])

    report = {
        "artifact_a": str(args.artifact_a),
        "artifact_b": str(args.artifact_b) if args.artifact_b else None,
        "num_subjects": int(subjects.shape[0]),
        "num_samples": int(first["labels"].shape[0]),
        "num_bootstrap": args.num_bootstrap,
        "bootstrap_attempts": attempts,
        "discarded_missing_class_resamples": (
            attempts - args.num_bootstrap
        ),
        "seed": args.seed,
        "classes": classes.tolist(),
        "point_estimate_a": metrics(
            first["labels"],
            first["logits"],
            classes,
            first.get("decision_threshold"),
        ),
        "bootstrap_a": {
            name: summarize(values) for name, values in first_samples.items()
        },
    }
    if second is not None:
        report["point_estimate_b"] = metrics(
            second["labels"],
            second["logits"],
            classes,
            second.get("decision_threshold"),
        )
        report["bootstrap_b"] = {
            name: summarize(values)
            for name, values in second_samples.items()
        }
        report["paired_delta_b_minus_a"] = {
            name: summarize(values)
            for name, values in delta_samples.items()
        }

    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
