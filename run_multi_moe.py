import os
import argparse
import pickle
import json
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import multiclass_metrics_fn

import linear_MoE
from linear_MoE import linear_MoEClassifier

import EEGclassifier
from EEGclassifier import EEGClassifierxr

from utils import (
    TUEVLoader,
    HARLoader,
    SEEDLoader,
    SEEDVLoader,
    tuev_subject_id,
)
from experiment_utils import (
    artifact_metadata,
    gate_diagnostics,
    save_experiment_result,
    save_prediction_artifact,
    seed_everything,
    source_snapshot_metadata,
)

import torch.distributed as dist


FINAL_LOCK_FIELDS = (
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


def _verify_read_only_lock(lock_path):
    if lock_path.stat().st_mode & 0o222:
        raise ValueError(
            "final config lock is writable; recreate it with the lock tool"
        )
    ledger_path = lock_path.parent / "lock_ledger.jsonl"
    if not ledger_path.is_file():
        raise ValueError("final config lock ledger is missing")
    lock_hash = artifact_metadata(lock_path)["sha256"]
    ledger_records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not any(
        record.get("lock", {}).get("sha256") == lock_hash
        for record in ledger_records
    ):
        raise ValueError("final config lock hash is absent from its ledger")


def verify_evaluation_lock(args, current_source_snapshot):
    """Fail before training if a test-evaluation mode is not predeclared."""
    if args.evaluation_mode == "validation_only":
        return None
    if args.evaluation_mode == "smoke_test":
        limits = (
            args.limit_train_batches,
            args.limit_val_batches,
            args.limit_test_batches,
        )
        if all(limit == 1.0 for limit in limits):
            raise ValueError(
                "smoke_test requires at least one limited-batch fraction"
            )
        if "smoke" not in Path(args.result_dir).parts:
            raise ValueError("smoke_test result_dir must be under results/smoke")
        return {
            "mode": "smoke_test",
            "paper_evidence": False,
            "test_access_permitted": False,
        }

    if not args.final_config_lock:
        raise ValueError(
            "final_test requires --final_config_lock created after "
            "validation-only model selection"
        )
    if any(
        limit != 1.0
        for limit in (
            args.limit_train_batches,
            args.limit_val_batches,
            args.limit_test_batches,
        )
    ):
        raise ValueError("final_test requires all batch limits to equal 1.0")
    lock_path = Path(args.final_config_lock)
    _verify_read_only_lock(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("lock_version") != 2:
        raise ValueError("final config lock must use lock_version 2")
    locked_config = lock.get("locked_config")
    if not isinstance(locked_config, dict):
        raise ValueError("final config lock is missing locked_config")
    missing = [
        field for field in FINAL_LOCK_FIELDS if field not in locked_config
    ]
    if missing:
        raise ValueError(f"final config lock is missing fields: {missing}")
    mismatches = {
        field: {
            "locked": locked_config[field],
            "requested": getattr(args, field),
        }
        for field in FINAL_LOCK_FIELDS
        if locked_config[field] != getattr(args, field)
    }
    if mismatches:
        raise ValueError(
            "requested final-test config differs from lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    validation_evidence = lock.get("validation_evidence")
    if not validation_evidence:
        raise ValueError("final config lock lacks validation_evidence")
    manifest = artifact_metadata(args.tuev_split_manifest)
    if lock.get("tuev_manifest_sha256") != manifest["sha256"]:
        raise ValueError("TUEV manifest hash differs from final config lock")
    mapping_summary = artifact_metadata(args.tuev_file_manifest_summary)
    if (
        lock.get("tuev_mapping_summary_sha256")
        != mapping_summary["sha256"]
    ):
        raise ValueError(
            "TUEV file-mapping summary differs from final config lock"
        )
    locked_snapshot = lock.get("source_snapshot_at_training_start", {})
    if (
        locked_snapshot.get("source_snapshot_sha256")
        != current_source_snapshot.get("source_snapshot_sha256")
    ):
        raise ValueError(
            "current executable source differs from the validation-training "
            "snapshot locked for final evaluation"
        )
    checkpoint = lock.get("selected_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("final config lock lacks selected_checkpoint")
    checkpoint_record = artifact_metadata(checkpoint.get("path"))
    if checkpoint_record["sha256"] != checkpoint.get("sha256"):
        raise ValueError("locked checkpoint hash mismatch")
    if args.pretrain_model_path:
        pretrain_record = artifact_metadata(args.pretrain_model_path)
        if pretrain_record != lock.get("pretrain_artifact"):
            raise ValueError("pretraining artifact differs from final lock")
    return {
        "mode": "final_test",
        "lock": artifact_metadata(lock_path),
        "validation_evidence": validation_evidence,
        "selected_checkpoint": checkpoint_record,
        "source_snapshot": locked_snapshot,
        "routing_analysis": lock.get("routing_analysis"),
    }


def verify_tuev_file_mapping(args):
    summary_path = Path(args.tuev_file_manifest_summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "subject_id_parser_version": 2,
        "processed_file_counts": {"train": 83932, "eval": 29421},
        "subject_counts": {"train": 290, "eval": 80},
    }
    mismatches = {
        key: {"expected": value, "found": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Frozen TUEV mapping summary mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    mapping_path = Path(summary["mapping_path"])
    mapping_record = artifact_metadata(mapping_path)
    if mapping_record["sha256"] != summary["mapping_sha256"]:
        raise RuntimeError("Frozen TUEV mapping JSONL hash mismatch")
    return {
        "summary": artifact_metadata(summary_path),
        "mapping": mapping_record,
        "join_invariant": summary["join_invariant"],
    }

# Implementation note.
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from datetime import datetime

# Implementation note.
class LossPlotCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        print("--- LossPlotCallback: on_train_end called ---")
        # Implementation note.
        save_dir = Path("/home/xr/LBT/metrics")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Implementation note.
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Implementation note.
        train_losses = pl_module.train_loss_history
        val_losses = pl_module.val_loss_history
        
        # Implementation note.
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
        if val_losses:  # Implementation note.
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
        
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Implementation note.
        save_path = save_dir / f'loss_vs_epochs_{current_time}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {save_path}")

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        # Implementation note.
        self.train_loss_history = []
        self.val_loss_history = []
        self.validation_logits = None
        self.validation_labels = None
        self.test_logits = None
        self.test_labels = None
        self.test_expert_logits = None
        self.test_gate_weights = None
        self.test_channel_expert_scores = None
        self.test_permuted_gate_logits = None
        self.test_gate_permutation = None
        self.test_causal_diagnostics = {}

    def on_train_epoch_start(self):
        if self.args.transfer_mode == "frozen_linear":
            self.model.moe.eval()

    def training_step(self, batch, batch_idx):
        X, y = batch
        prod = self.model(X)
        task_loss = nn.CrossEntropyLoss()(prod, y)
        routing_aux_loss = getattr(self.model, "routing_aux_loss", None)
        loss = task_loss
        if routing_aux_loss is not None:
            loss = loss + self.args.router_aux_loss_coef * routing_aux_loss
            self.log("train_router_aux_loss", routing_aux_loss)
        self.log("train_task_loss", task_loss)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Implementation note.
        avg_train_loss = self.trainer.callback_metrics.get("train_loss").item()
        self.train_loss_history.append(avg_train_loss)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            val_loss = nn.CrossEntropyLoss()(convScore, y)
            self.log("val_loss", val_loss)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def on_validation_epoch_end(self):
        # Implementation note.
        avg_val_loss = self.trainer.callback_metrics.get("val_loss").item()
        self.val_loss_history.append(avg_val_loss)

    def validation_epoch_end(self, val_step_outputs):
        result = []
        gt = np.array([])
        for out in val_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        logits = np.concatenate(result, axis=0)
        self.validation_logits = logits
        self.validation_labels = gt.astype(np.int64)
        result = multiclass_metrics_fn(
            gt, logits, metrics=["accuracy", "cohen_kappa", "f1_weighted","balanced_accuracy"]
        )
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("val_f1", result["f1_weighted"], sync_dist=True)
        self.log("balanced_accuracy",result["balanced_accuracy"],sync_dist=True)
        print(result)


    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            if hasattr(self.model, "forward_with_expert_details"):
                details = self.model.forward_with_expert_details(X)
                return {
                    "logits": details["logits"].cpu().numpy(),
                    "labels": y.cpu().numpy(),
                    "expert_logits": details["expert_logits"].cpu().numpy(),
                    "gate_weights": details["gate_weights"].cpu().numpy(),
                    "channel_expert_scores": (
                        details["channel_expert_scores"].cpu().numpy()
                    ),
                }
            return {
                "logits": self.model(X).cpu().numpy(),
                "labels": y.cpu().numpy(),
            }

    def test_epoch_end(self, test_step_outputs):
        result = []
        gt = []
        expert_logits = []
        gate_weights = []
        channel_expert_scores = []
        for out in test_step_outputs:
            result.append(out["logits"])
            gt.append(out["labels"])
            if "expert_logits" in out:
                expert_logits.append(out["expert_logits"])
                gate_weights.append(out["gate_weights"])
                channel_expert_scores.append(
                    out["channel_expert_scores"]
                )

        logits = np.concatenate(result, axis=0)
        gt = np.concatenate(gt, axis=0)
        self.test_logits = logits
        self.test_labels = gt.astype(np.int64)
        if expert_logits:
            self.test_expert_logits = np.concatenate(expert_logits, axis=0)
            self.test_gate_weights = np.concatenate(gate_weights, axis=0)
            self.test_channel_expert_scores = np.concatenate(
                channel_expert_scores, axis=0
            )
            permutation_rng = np.random.RandomState(self.args.seed + 1701)
            self.test_gate_permutation = np.arange(gt.shape[0])
            # Sattolo's algorithm produces one cycle and therefore no fixed
            # points when at least two held-out samples are available.
            for index in range(gt.shape[0] - 1, 0, -1):
                swap_index = permutation_rng.randint(0, index)
                (
                    self.test_gate_permutation[index],
                    self.test_gate_permutation[swap_index],
                ) = (
                    self.test_gate_permutation[swap_index],
                    self.test_gate_permutation[index],
                )
            permuted_gate_weights = self.test_gate_weights[
                self.test_gate_permutation
            ]
            classifier_bias = (
                self.model.classifier[-1].bias.detach().cpu().numpy()
            )
            self.test_permuted_gate_logits = (
                (
                    self.test_channel_expert_scores
                    * permuted_gate_weights[..., np.newaxis]
                )
                .sum(axis=2)
                .mean(axis=1)
                + classifier_bias
            )
            permuted_metrics = multiclass_metrics_fn(
                gt,
                self.test_permuted_gate_logits,
                metrics=[
                    "accuracy",
                    "cohen_kappa",
                    "f1_weighted",
                    "balanced_accuracy",
                ],
            )
            sample_indices = np.arange(gt.shape[0])
            shifted = logits - logits.max(axis=1, keepdims=True)
            mixture_probabilities = np.exp(shifted)
            mixture_probabilities /= mixture_probabilities.sum(
                axis=1, keepdims=True
            )
            expert_shifted = self.test_expert_logits - (
                self.test_expert_logits.max(axis=2, keepdims=True)
            )
            expert_probabilities = np.exp(expert_shifted)
            expert_probabilities /= expert_probabilities.sum(
                axis=2, keepdims=True
            )
            correct_class_probabilities = expert_probabilities[
                sample_indices, :, self.test_labels
            ]
            best_single_expert = correct_class_probabilities.argmax(axis=1)
            best_single_expert_logits = self.test_expert_logits[
                sample_indices, best_single_expert
            ]
            best_single_expert_metrics = multiclass_metrics_fn(
                gt,
                best_single_expert_logits,
                metrics=[
                    "accuracy",
                    "cohen_kappa",
                    "f1_weighted",
                    "balanced_accuracy",
                ],
            )
            oracle_correct_probability = expert_probabilities[
                sample_indices, best_single_expert, self.test_labels
            ]
            mixture_correct_probability = mixture_probabilities[
                sample_indices, self.test_labels
            ]
            safe_gate = np.clip(self.test_gate_weights, 1e-12, None)
            gate_entropy = -(safe_gate * np.log(safe_gate)).sum(axis=-1)
            gate_mean_by_label = {
                str(int(label)): self.test_gate_weights[
                    self.test_labels == label
                ].mean(axis=(0, 1)).tolist()
                for label in np.unique(self.test_labels)
            }
            self.test_causal_diagnostics = {
                "permuted_gate_metrics": permuted_metrics,
                "best_single_expert_reference_metrics": (
                    best_single_expert_metrics
                ),
                "best_single_expert_minus_mixture_correct_probability_mean": float(
                    (
                        oracle_correct_probability
                        - mixture_correct_probability
                    ).mean()
                ),
                "best_single_expert_selection_rate": (
                    np.bincount(
                        best_single_expert,
                        minlength=self.test_expert_logits.shape[1],
                    )
                    / best_single_expert.shape[0]
                ).tolist(),
                "gate_mean_by_label": gate_mean_by_label,
                "gate_entropy_mean_full_test": float(gate_entropy.mean()),
                "gate_entropy_std_full_test": float(gate_entropy.std()),
                "gate_permutation_seed": int(self.args.seed + 1701),
                "gate_permutation_fixed_points": int(
                    (
                        self.test_gate_permutation
                        == np.arange(gt.shape[0])
                    ).sum()
                ),
                "note": (
                    "Gate permutation is performed once across the complete "
                    "held-out sample set, after all batches are collected. "
                    "The best-single-expert reference maximizes true-class "
                    "probability using held-out labels. It is diagnostic, "
                    "not deployable and not a formal upper bound. The saved "
                    "artifact supports the predeclared multi-permutation "
                    "analysis."
                ),
            }
        result = multiclass_metrics_fn(
            gt, logits, metrics=["accuracy", "cohen_kappa", "f1_weighted","balanced_accuracy"]
        )
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("test_f1", result["f1_weighted"], sync_dist=True)
        self.log("balanced_accuracy",result["balanced_accuracy"],sync_dist=True)
        return result

    def configure_optimizers(self):
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters remain")
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_TUEV_dataloader(args, include_test=False):
    #substract
    root = args.tuev_root

    # Implementation note.
    train_files = sorted(
        filename
        for filename in os.listdir(os.path.join(root, "processed_train"))
        if filename.endswith(".pkl")
    )
    all_train_sub = {
        tuev_subject_id(filename, "train") for filename in train_files
    }
    train_sub = sorted(all_train_sub)
    print("train sub", len(train_sub))
    manifest_path = Path(args.tuev_split_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "TUEV split manifest must be audited and frozen before training: "
            f"{manifest_path}"
        )
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_sub = set(split_manifest["val_subjects"])
    train_sub = set(split_manifest["train_subjects"])
    test_sub = set(split_manifest.get("test_subjects", []))
    if train_sub | val_sub != all_train_sub:
        raise RuntimeError(
            "Persisted TUEV train/validation subjects do not match "
            "processed_train"
        )
    if split_manifest.get("subject_id_parser_version") != 2:
        raise RuntimeError(
            "Persisted TUEV manifest uses an obsolete subject parser; "
            "repair and freeze it before training"
        )
    test_files = None
    if include_test:
        test_files = sorted(
            filename
            for filename in os.listdir(os.path.join(root, "processed_eval"))
            if filename.endswith(".pkl")
        )
        actual_test_sub = {
            tuev_subject_id(filename, "eval") for filename in test_files
        }
        if actual_test_sub != test_sub:
            raise RuntimeError(
                "Persisted TUEV evaluation subjects do not match "
                "processed_eval"
            )

    if train_sub & val_sub:
        raise RuntimeError("TUEV train/validation subject overlap detected")
    if train_sub & test_sub or val_sub & test_sub:
        raise RuntimeError("TUEV evaluation subjects overlap train/validation")
    val_files = [
        filename
        for filename in train_files
        if tuev_subject_id(filename, "train") in val_sub
    ]
    train_files = [
        filename
        for filename in train_files
        if tuev_subject_id(filename, "train") in train_sub
    ]
    full_train_files = list(train_files)
    if args.label_fraction < 1.0:
        subset_path = Path(args.label_subset_manifest)
        if not subset_path.is_file():
            raise FileNotFoundError(
                "Frozen TUEV label-fraction manifest is required: "
                f"{subset_path}"
            )
        subset_manifest = json.loads(
            subset_path.read_text(encoding="utf-8")
        )
        if subset_manifest.get("artifact_version") != 1:
            raise ValueError("Unsupported TUEV label subset manifest")
        if subset_manifest.get("dataset") != "TUEV":
            raise ValueError("Label subset manifest is not for TUEV")
        digest = hashlib.sha256()
        for filename in sorted(full_train_files):
            digest.update(filename.encode("utf-8"))
            digest.update(b"\n")
        if digest.hexdigest() != subset_manifest.get(
            "full_train_file_inventory_sha256"
        ):
            raise RuntimeError(
                "TUEV training inventory differs from label subset manifest"
            )
        split_record = artifact_metadata(args.tuev_split_manifest)
        if split_record["sha256"] != subset_manifest.get(
            "tuev_split_manifest_sha256"
        ):
            raise RuntimeError(
                "TUEV split manifest differs from label subset manifest"
            )
        fraction_key = f"{args.label_fraction:g}"
        subset = subset_manifest.get("subsets", {}).get(fraction_key)
        if not isinstance(subset, dict):
            raise ValueError(
                f"Label subset manifest lacks fraction {fraction_key}"
            )
        train_files = list(subset.get("files", []))
        if len(train_files) != subset.get("file_count"):
            raise RuntimeError("TUEV label subset file count mismatch")
        if not set(train_files).issubset(set(full_train_files)):
            raise RuntimeError("TUEV label subset contains a non-training file")
        args.effective_label_subset_manifest = str(subset_path)
    else:
        args.effective_label_subset_manifest = ""

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = None
    if include_test:
        test_loader = torch.utils.data.DataLoader(
            TUEVLoader(
                os.path.join(root, "processed_eval"),
                test_files,
                args.sampling_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    print(
        len(train_files),
        len(val_files),
        None if test_files is None else len(test_files),
    )
    print(
        len(train_loader),
        len(val_loader),
        None if test_loader is None else len(test_loader),
    )
    return train_loader, test_loader, val_loader

def prepare_SEED_dataloader(args):
    root = args.seed_root

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))


    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            os.path.join(
                root, "train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            os.path.join(
                root, "test"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            os.path.join(
                root, "val"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_SEEDV_dataloader(args):
    root = args.seedv_root

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))


    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        SEEDVLoader(
            os.path.join(
                root, "train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        SEEDVLoader(
            os.path.join(
                root, "test"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        SEEDVLoader(
            os.path.join(
                root, "val"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

from model.dwmoespace import dwmoespace
from model.dwmoespace_stack import dwmoespace_stack
from model.dwmoespace_newgate import dwmoespace_newgate
from model.dwmoespace_sparse_topk import dwmoespace_sparse_topk
from model.dw import dw

def supervised(args):
    args.resolved_checkpoint_metric = (
        "val_cohen"
        if args.checkpoint_metric == "auto"
        else args.checkpoint_metric
    )
    source_snapshot_at_start = source_snapshot_metadata()
    seed_everything(args.seed, args.deterministic)
    pl.seed_everything(args.seed, workers=True)
    evaluation_lock = verify_evaluation_lock(args, source_snapshot_at_start)
    tuev_mapping = verify_tuev_file_mapping(args)
    # get data loaders
    if args.dataset == "TUEV":
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(
            args,
            include_test=args.evaluation_mode == "final_test",
        )
    elif args.dataset == "SEED":
        train_loader, test_loader, val_loader = prepare_SEED_dataloader(args)
    elif args.dataset == "SEEDV":
        train_loader, test_loader, val_loader = prepare_SEEDV_dataloader(args)
    else:
        raise NotImplementedError

    # define the model
    if args.model == "linear_MoE":
        model = linear_MoEClassifier(
            batch_size = args.batch_size,
            feature_in = args.feature_in,
            feature_out = args.feature_out,
            expert_number = args.expert_number,
            n_classes=args.n_classes
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):  # Implementation note.
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
   
    elif args.model == "eegclassifier":
        model = EEGClassifierxr(
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):  # Implementation note.
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    elif args.model == "dwmoespace":
        model = dwmoespace(num_classes=args.n_classes,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):  # Implementation note.
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    elif args.model == "dwmoespace_stack":
        model = dwmoespace_stack(num_classes=args.n_classes,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):  # Implementation note.
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    elif args.model == "dw":
        model = dw(num_classes=args.n_classes,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):  # Implementation note.
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    
    # elif args.model == "dwmoespace_newgate":
    #     model = dwmoespace_newgate(num_classes=args.n_classes,
    #     )
    # Implementation note.
    # Implementation note.
    #         state_dict = torch.load(args.pretrain_model_path)["state_dict"]
            
    # Implementation note.
    #         moe_encoder_state_dict = {
    #             k.replace("model.moe.", ""): v
    #             for k, v in state_dict.items()
    # Implementation note.
    #         }
    # Implementation note.
    #         model.moe.load_state_dict(moe_encoder_state_dict, strict=True)

    # Implementation note.
    #         # freeze_before_moe = True
    #         # for name, param in model.lbt.named_parameters():
    # Implementation note.
    #         #         freeze_before_moe = False
    #         #     param.requires_grad = not freeze_before_moe

    elif args.model == "dwmoespace_newgate":
        model = dwmoespace_newgate(
            num_classes=args.n_classes,
            expert_output_dim=args.feature_out,
            gate_type=args.gate_type,
            gate_hidden_dim=args.gate_hidden_dim,
            expert_axis=args.expert_axis,
            relation_mode=args.relation_mode,
            norm_type=args.norm_type,
            gate_descriptor_source=args.gate_descriptor_source,
            match_dynamic_budget=args.match_dynamic_budget,
            expert_kernels=args.expert_kernels,
        )
        
        if args.pretrain_model_path and (args.sampling_rate == 200):  # Implementation note.
            state_dict = torch.load(args.pretrain_model_path)["state_dict"]
            
            # Implementation note.
            moe_encoder_state_dict = {
                k.replace("model.moe.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.moe.")  # Implementation note.
            }
            
            # Implementation note.
            model.moe.load_state_dict(moe_encoder_state_dict, strict=True)

        if args.transfer_mode == "frozen_linear":
            for parameter in model.moe.parameters():
                parameter.requires_grad_(False)
            model.moe.eval()
            
        # Implementation note.
        # Implementation note.
        #     for name, param in model.moe.temporal_conv.named_parameters():
        # Implementation note.
        # Implementation note.
        
        # Implementation note.
        # Implementation note.
        # trainable_params = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         trainable_params.append(name)
        # Implementation note.
        
        # if not trainable_params:
        # Implementation note.
        # else:
        # Implementation note.

    elif args.model == "dwmoespace_sparse_topk":
        if args.pretrain_model_path:
            raise ValueError(
                "Sparse Top-K experiments are trained from scratch; pretrained "
                "loading requires a separately verified checkpoint adapter"
            )
        if args.transfer_mode != "full_finetune":
            raise ValueError(
                "Sparse Top-K experiments currently support full_finetune only"
            )
        model = dwmoespace_sparse_topk(
            num_classes=args.n_classes,
            expert_output_dim=args.feature_out,
            router_hidden_dim=args.gate_hidden_dim,
            expert_kernels=args.expert_kernels,
            top_k=args.top_k,
            router_temperature=args.router_temperature,
            norm_type=args.norm_type,
        )

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)

    # logger and callbacks
    version = (
        f"{args.dataset}-{args.model}-{args.expert_axis}-{args.norm_type}-"
        f"{args.relation_mode}-{args.gate_descriptor_source}-"
        f"{args.gate_type}-topk{args.top_k}-"
        f"k{args.expert_kernels.replace(',', '-')}-"
        f"d{args.feature_out}-{args.evaluation_mode}-seed{args.seed}-"
        f"{args.transfer_mode}-labels{args.label_fraction:g}-"
        f"{args.lr}-{args.batch_size}-{args.sampling_rate}-"
        f"{args.token_size}-{args.hop_length}-{args.run_id or 'auto'}"
    )
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    checkpoint_metric = args.resolved_checkpoint_metric
    early_stop_callback = EarlyStopping(
        monitor=checkpoint_metric, patience=5, verbose=False, mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_metric,
        mode="max",
        save_top_k=1,
        filename="{epoch:03d}-{" + checkpoint_metric + ":.4f}",
    )

    is_final_test = args.evaluation_mode == "final_test"
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        strategy="auto",
        auto_select_gpus=True,
        benchmark=not args.deterministic,
        deterministic="warn" if args.deterministic else False,
        enable_checkpointing=not is_final_test,
        logger=logger,
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        callbacks=(
            [] if is_final_test else [checkpoint_callback, early_stop_callback]
        ),
    )

    checkpoint_record = None
    if not is_final_test:
        trainer.fit(
            lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        checkpoint_record = artifact_metadata(
            checkpoint_callback.best_model_path
        )
    else:
        checkpoint_record = evaluation_lock["selected_checkpoint"]
    manifest_record = artifact_metadata(args.tuev_split_manifest)
    diagnostics = {
        "evaluation_mode": args.evaluation_mode,
        "evaluation_lock": evaluation_lock,
        "best_checkpoint": checkpoint_record,
        "tuev_manifest": manifest_record,
        "tuev_file_mapping": tuev_mapping,
        "test_loader_constructed": test_loader is not None,
        "source_snapshot_at_start": source_snapshot_at_start,
        "transfer_mode": args.transfer_mode,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in lightning_model.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in lightning_model.parameters()
            if not parameter.requires_grad
        ),
        "label_fraction": args.label_fraction,
        "label_subset_manifest": (
            artifact_metadata(args.effective_label_subset_manifest)
            if args.effective_label_subset_manifest
            else None
        ),
    }
    if not is_final_test:
        args.result_scope = (
            "validation_selection_only"
            if args.evaluation_mode == "validation_only"
            else "non_evidentiary_smoke"
        )
        if hasattr(lightning_model.model.moe, "reset_routing_stats"):
            lightning_model.model.moe.reset_routing_stats()
        pretrain_result = trainer.validate(
            model=lightning_model,
            ckpt_path="best",
            dataloaders=val_loader,
        )[0]
        diagnostics["test_set_accessed"] = False
        validation_files = list(val_loader.dataset.files)
        diagnostics["validation_prediction_artifact"] = (
            save_prediction_artifact(
                args,
                lightning_model.validation_logits,
                lightning_model.validation_labels,
                sample_ids=validation_files,
                subject_ids=[
                    tuev_subject_id(filename, "train")
                    for filename in validation_files
                ],
            )
        )
    else:
        args.result_scope = "locked_final_test"
        diagnostics["test_set_accessed"] = True
        if hasattr(lightning_model.model.moe, "reset_routing_stats"):
            lightning_model.model.moe.reset_routing_stats()
        pretrain_result = trainer.test(
            model=lightning_model,
            ckpt_path=checkpoint_record["path"],
            dataloaders=test_loader,
        )[0]
        test_files = list(test_loader.dataset.files)
        prediction_record = save_prediction_artifact(
            args,
            lightning_model.test_logits,
            lightning_model.test_labels,
            sample_ids=test_files,
            subject_ids=[
                tuev_subject_id(filename, "eval") for filename in test_files
            ],
            extra_arrays=(
                {
                    "expert_logits": lightning_model.test_expert_logits,
                    "gate_weights": lightning_model.test_gate_weights,
                    "channel_expert_scores": (
                        lightning_model.test_channel_expert_scores
                    ),
                    "permuted_gate_logits": (
                        lightning_model.test_permuted_gate_logits
                    ),
                    "gate_permutation": (
                        lightning_model.test_gate_permutation
                    ),
                }
                if lightning_model.test_expert_logits is not None
                else None
            ),
        )
        diagnostics["last_test_batch_gate_summary"] = gate_diagnostics(
            lightning_model.model
        )
        diagnostics["prediction_artifact"] = prediction_record
        diagnostics["causal_routing"] = (
            lightning_model.test_causal_diagnostics
        )
    if hasattr(lightning_model.model.moe, "routing_summary"):
        diagnostics["sparse_routing"] = (
            lightning_model.model.moe.routing_summary()
        )
    print(pretrain_result)
    result_path = save_experiment_result(
        args,
        pretrain_result,
        extra=diagnostics,
    )
    print(f"result_json: {result_path}")
    if dist.is_initialized():
        dist.destroy_process_group()

#python run_multi_moe.py --batch_size 512 --dataset TUEV --model dwmoespace --n_classes 6
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,default=128, help="batch size")
    parser.add_argument("--num_workers", type=int,default=8, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUEV", help="dataset")
    parser.add_argument("--expert_number", type=int, default=4, help="number of experts")
    parser.add_argument("--model", type=str, default="dwmoespace_newgate", help="which supervised model to use")
    parser.add_argument("--feature_in", type=int, default=16000, help="number of input channels")
    parser.add_argument("--feature_out",type=int,default=299,help="number of output token")
    parser.add_argument("--in_channels", type=int, default=12, help="number of input channels")
    parser.add_argument("--sample_length", type=float, default=10, help="length (s) of sample")
    parser.add_argument("--n_classes", type=int, default=6, help="number of output classes")
    parser.add_argument("--sampling_rate", type=int, default=200, help="sampling rate (r)")
    parser.add_argument("--token_size", type=int,default=200, help="token size (t)")
    parser.add_argument("--hop_length", type=int, default=100, help="token hop length (t - p)")
    parser.add_argument("--pretrain_model_path", type=str, default="", help="pretrained model path")
    parser.add_argument(
        "--transfer_mode",
        choices=["full_finetune", "frozen_linear"],
        default="full_finetune",
    )
    parser.add_argument(
        "--label_fraction",
        type=float,
        choices=[0.01, 0.1, 1.0],
        default=1.0,
    )
    parser.add_argument(
        "--label_subset_manifest",
        type=str,
        default="results/manifests/tuev_label_subsets_seed2026.json",
    )
    parser.add_argument("--top_k",type=int,default=3,help="top k expert")
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--gate_type", choices=["uniform", "static", "input"], default="uniform")
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--expert_axis", choices=["legacy", "temporal"], default="temporal")
    parser.add_argument(
        "--norm_type",
        choices=["auto", "legacy_global", "local_filter"],
        default="local_filter",
    )
    parser.add_argument(
        "--gate_descriptor_source",
        choices=["raw", "encoded"],
        default="raw",
    )
    parser.add_argument("--match_dynamic_budget", action="store_true")
    parser.add_argument(
        "--expert_kernels",
        type=str,
        default="7",
        help="comma-separated odd temporal kernels; temporal expert axis only",
    )
    parser.add_argument(
        "--relation_mode",
        choices=[
            "none",
            "attention",
            "static",
            "dynamic",
            "dynamic_normalized",
            "spatial1x1",
            "static_conditioned_matched",
            "spatial1x1_conditioned_matched",
            "dynamic_normalized_attention",
        ],
        default="none",
    )
    parser.add_argument(
        "--checkpoint_metric",
        choices=["auto", "val_cohen", "balanced_accuracy"],
        default="auto",
        help="auto uses validation Cohen's kappa for TUEV",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="results/gate_ablation")
    parser.add_argument(
        "--evaluation_mode",
        choices=["validation_only", "smoke_test", "final_test"],
        default="validation_only",
        help=(
            "validation_only never calls the test loader; final_test requires "
            "a matching immutable config lock"
        ),
    )
    parser.add_argument("--final_config_lock", type=str, default="")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument(
        "--tuev_root",
        type=str,
        default="/home/dataset/tuh_eeg/tuh_eeg_events/tuh_eeg_events/v2.0.1/edf",
    )
    parser.add_argument(
        "--tuev_split_manifest",
        type=str,
        default="results/manifests/tuev_subject_split.json",
    )
    parser.add_argument(
        "--tuev_file_manifest_summary",
        type=str,
        default=(
            "results/manifests/"
            "tuev_processed_file_mapping_summary.json"
        ),
    )
    parser.add_argument("--seed_root", type=str, default="/home/xr/SEED_process16/SEED/processed")
    parser.add_argument("--seedv_root", type=str, default="/home/xr/SEED_process16/SEED-V/processed")
    args = parser.parse_args()
    print(args)

    supervised(args)


