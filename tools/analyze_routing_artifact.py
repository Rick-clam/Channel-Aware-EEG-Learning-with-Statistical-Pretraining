"""Causal routing diagnostics from a locked final-test prediction artifact.

This analysis is intentionally post hoc with respect to model fitting, but its
algorithm and random seeds must be frozen in the final configuration lock before
the held-out test set is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)


REQUIRED_ARRAYS = {
    "logits",
    "labels",
    "subject_ids",
    "expert_logits",
    "gate_weights",
    "channel_expert_scores",
}


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def true_class_nll(logits, labels):
    probabilities = softmax(logits)
    indices = np.arange(labels.shape[0])
    return -np.log(
        np.clip(probabilities[indices, labels], 1e-12, 1.0)
    )


def classification_metrics(labels, logits):
    predictions = logits.argmax(axis=1)
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mean_nll": float(true_class_nll(logits, labels).mean()),
    }


def sattolo_permutation(length, rng):
    if length < 2:
        return np.arange(length)
    permutation = np.arange(length)
    for index in range(length - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        permutation[index], permutation[swap_index] = (
            permutation[swap_index],
            permutation[index],
        )
    return permutation


def subject_blocked_permutation(subject_ids, rng):
    """Derange samples independently within each subject."""
    permutation = np.arange(subject_ids.shape[0])
    for subject in np.unique(subject_ids):
        indices = np.flatnonzero(subject_ids == subject)
        if indices.shape[0] >= 2:
            permutation[indices] = indices[
                sattolo_permutation(indices.shape[0], rng)
            ]
    return permutation


def reconstruct_logits(scores, gate_weights, classifier_bias):
    return (
        (scores * gate_weights[..., np.newaxis])
        .sum(axis=2)
        .mean(axis=1)
        + classifier_bias
    )


def ranks(values):
    order = np.argsort(values, axis=1, kind="mergesort")
    result = np.empty_like(order, dtype=np.float64)
    row_indices = np.arange(values.shape[0])[:, np.newaxis]
    result[row_indices, order] = np.arange(values.shape[1], dtype=float)
    return result


def rowwise_correlation(first, second):
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        (first * first).sum(axis=1) * (second * second).sum(axis=1)
    )
    output = np.zeros(first.shape[0], dtype=np.float64)
    valid = denominator > 0
    output[valid] = (
        (first[valid] * second[valid]).sum(axis=1) / denominator[valid]
    )
    return output


def cluster_bootstrap_mean(values, subject_ids, count, rng):
    subjects = np.unique(subject_ids)
    indices = {
        subject: np.flatnonzero(subject_ids == subject)
        for subject in subjects
    }
    samples = np.empty(count, dtype=np.float64)
    for iteration in range(count):
        sampled = rng.choice(subjects, size=subjects.shape[0], replace=True)
        selected = np.concatenate([indices[subject] for subject in sampled])
        samples[iteration] = values[selected].mean()
    return {
        "point_estimate": float(values.mean()),
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_standard_error": float(samples.std(ddof=1)),
        "ci_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def permutation_analysis(
    scores,
    gate_weights,
    classifier_bias,
    labels,
    subject_ids,
    num_permutations,
    seed,
    blocked,
):
    original_nll = true_class_nll(
        reconstruct_logits(scores, gate_weights, classifier_bias),
        labels,
    )
    average_permuted_nll = np.zeros(labels.shape[0], dtype=np.float64)
    mean_deltas = np.empty(num_permutations, dtype=np.float64)
    fixed_points = np.empty(num_permutations, dtype=np.int64)
    for iteration in range(num_permutations):
        rng = np.random.default_rng(seed + iteration)
        if blocked:
            permutation = subject_blocked_permutation(subject_ids, rng)
        else:
            permutation = sattolo_permutation(labels.shape[0], rng)
        fixed_points[iteration] = int(
            (permutation == np.arange(labels.shape[0])).sum()
        )
        permuted_nll = true_class_nll(
            reconstruct_logits(
                scores,
                gate_weights[permutation],
                classifier_bias,
            ),
            labels,
        )
        average_permuted_nll += permuted_nll / num_permutations
        mean_deltas[iteration] = float(
            (permuted_nll - original_nll).mean()
        )
    sample_delta = average_permuted_nll - original_nll
    bootstrap_rng = np.random.default_rng(seed + 10_000_000)
    return {
        "scheme": (
            "independent within-subject Sattolo derangements"
            if blocked
            else "full-test Sattolo derangements"
        ),
        "num_permutations": int(num_permutations),
        "seed_start": int(seed),
        "fixed_points_min": int(fixed_points.min()),
        "fixed_points_max": int(fixed_points.max()),
        "mean_nll_delta_distribution": {
            "mean": float(mean_deltas.mean()),
            "standard_deviation": float(mean_deltas.std(ddof=1)),
            "ci_95": [
                float(np.quantile(mean_deltas, 0.025)),
                float(np.quantile(mean_deltas, 0.975)),
            ],
            "one_sided_empirical_p_delta_le_zero": float(
                (1 + (mean_deltas <= 0).sum())
                / (num_permutations + 1)
            ),
        },
        "subject_cluster_bootstrap_mean_sample_nll_delta": (
            cluster_bootstrap_mean(
                sample_delta,
                subject_ids,
                10_000,
                bootstrap_rng,
            )
        ),
    }


def competence_analysis(
    expert_logits,
    gate_weights,
    labels,
    subject_ids,
    num_permutations,
    seed,
):
    sample_indices = np.arange(labels.shape[0])
    expert_probabilities = softmax(expert_logits)
    true_probabilities = expert_probabilities[
        sample_indices[:, np.newaxis],
        np.arange(expert_logits.shape[1])[np.newaxis, :],
        labels[:, np.newaxis],
    ]
    competence = np.log(np.clip(true_probabilities, 1e-12, 1.0))
    mean_gate = gate_weights.mean(axis=1)
    gate_ranks = ranks(mean_gate)
    competence_ranks = ranks(competence)
    sample_association = rowwise_correlation(
        gate_ranks, competence_ranks
    )

    null_means = np.empty(num_permutations, dtype=np.float64)
    for iteration in range(num_permutations):
        rng = np.random.default_rng(seed + iteration)
        permuted = np.empty_like(competence_ranks)
        for sample in range(labels.shape[0]):
            permuted[sample] = competence_ranks[
                sample, rng.permutation(competence_ranks.shape[1])
            ]
        null_means[iteration] = rowwise_correlation(
            gate_ranks, permuted
        ).mean()

    bootstrap_rng = np.random.default_rng(seed + 10_000_000)
    observed = float(sample_association.mean())
    return {
        "measure": (
            "within-sample Spearman correlation between channel-averaged "
            "gate mass and expert log true-class probability"
        ),
        "subject_cluster_bootstrap": cluster_bootstrap_mean(
            sample_association,
            subject_ids,
            10_000,
            bootstrap_rng,
        ),
        "expert-label-permutation_null": {
            "num_permutations": int(num_permutations),
            "seed_start": int(seed),
            "mean": float(null_means.mean()),
            "ci_95": [
                float(np.quantile(null_means, 0.025)),
                float(np.quantile(null_means, 0.975)),
            ],
            "one_sided_empirical_p_null_ge_observed": float(
                (1 + (null_means >= observed).sum())
                / (num_permutations + 1)
            ),
        },
    }


def best_single_expert_reference(expert_logits, mixture_logits, labels):
    sample_indices = np.arange(labels.shape[0])
    probabilities = softmax(expert_logits)
    true_probabilities = probabilities[
        sample_indices[:, np.newaxis],
        np.arange(expert_logits.shape[1])[np.newaxis, :],
        labels[:, np.newaxis],
    ]
    selection = true_probabilities.argmax(axis=1)
    selected_logits = expert_logits[sample_indices, selection]
    selected_nll = true_class_nll(selected_logits, labels)
    mixture_nll = true_class_nll(mixture_logits, labels)
    return {
        "definition": (
            "Per sample, select the single expert with maximum true-class "
            "probability (minimum cross-entropy). This label-informed "
            "reference is not deployable and is not a formal upper bound."
        ),
        "metrics": classification_metrics(labels, selected_logits),
        "selection_rate": (
            np.bincount(selection, minlength=expert_logits.shape[1])
            / selection.shape[0]
        ).tolist(),
        "mixture_minus_reference_nll_mean": float(
            (mixture_nll - selected_nll).mean()
        ),
        "reference_minus_mixture_true_probability_mean": float(
            (
                true_probabilities[sample_indices, selection]
                - softmax(mixture_logits)[sample_indices, labels]
            ).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--num_permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.artifact, allow_pickle=False) as artifact:
        missing = REQUIRED_ARRAYS - set(artifact.files)
        if missing:
            raise ValueError(
                f"{args.artifact} is missing arrays: {sorted(missing)}"
            )
        arrays = {
            name: artifact[name]
            for name in REQUIRED_ARRAYS
        }

    logits = arrays["logits"].astype(np.float64)
    labels = arrays["labels"].astype(np.int64)
    subject_ids = arrays["subject_ids"]
    expert_logits = arrays["expert_logits"].astype(np.float64)
    gate_weights = arrays["gate_weights"].astype(np.float64)
    scores = arrays["channel_expert_scores"].astype(np.float64)
    if not (
        logits.shape[0]
        == labels.shape[0]
        == subject_ids.shape[0]
        == expert_logits.shape[0]
        == gate_weights.shape[0]
        == scores.shape[0]
    ):
        raise ValueError("artifact arrays do not have the same sample count")

    weighted_scores = (
        (scores * gate_weights[..., np.newaxis])
        .sum(axis=2)
        .mean(axis=1)
    )
    bias_per_sample = logits - weighted_scores
    classifier_bias = bias_per_sample.mean(axis=0)
    reconstruction = weighted_scores + classifier_bias
    reconstruction_error = float(np.max(np.abs(reconstruction - logits)))
    if reconstruction_error > 1e-4:
        raise ValueError(
            "channel-expert scores do not reconstruct logits: "
            f"max_abs_error={reconstruction_error}"
        )

    report = {
        "artifact": str(args.artifact),
        "num_samples": int(labels.shape[0]),
        "num_subjects": int(np.unique(subject_ids).shape[0]),
        "num_experts": int(expert_logits.shape[1]),
        "num_permutations": int(args.num_permutations),
        "seed": int(args.seed),
        "base_metrics": classification_metrics(labels, logits),
        "logit_reconstruction_max_abs_error": reconstruction_error,
        "global_gate_permutation": permutation_analysis(
            scores,
            gate_weights,
            classifier_bias,
            labels,
            subject_ids,
            args.num_permutations,
            args.seed + 100_000,
            blocked=False,
        ),
        "within_subject_gate_permutation": permutation_analysis(
            scores,
            gate_weights,
            classifier_bias,
            labels,
            subject_ids,
            args.num_permutations,
            args.seed + 200_000,
            blocked=True,
        ),
        "gate_competence_association": competence_analysis(
            expert_logits,
            gate_weights,
            labels,
            subject_ids,
            args.num_permutations,
            args.seed + 300_000,
        ),
        "best_single_expert_reference": best_single_expert_reference(
            expert_logits,
            logits,
            labels,
        ),
    }

    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
