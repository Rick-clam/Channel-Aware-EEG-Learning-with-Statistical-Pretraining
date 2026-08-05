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
from pyhealth.metrics import binary_metrics_fn

import linear_MoE
from linear_MoE import linear_MoEClassifier

import EEGclassifier
from EEGclassifier import EEGClassifierxr

from torch.utils.data import IterableDataset

from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE
from experiment_utils import (
    artifact_metadata,
    gate_diagnostics,
    save_experiment_result,
    save_prediction_artifact,
    seed_everything,
    source_snapshot_metadata,
)
from chbmit_protocol import (
    canonical_chbmit_patient_id,
    enumerate_chbmit_windows,
    relative_inventory_sha256,
)


BINARY_FINAL_LOCK_FIELDS = (
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


def binary_subject_id(filename, dataset):
    """Return the subject component used by the persisted binary splits."""
    if dataset == "CHB_MIT":
        return canonical_chbmit_patient_id(filename)
    subject = os.path.basename(filename).split("_")[0]
    if not subject:
        raise ValueError(f"Cannot parse {dataset} subject from {filename}")
    return subject


def file_inventory_sha256(files):
    digest = hashlib.sha256()
    for filename in sorted(files):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _binary_manifest_path(args):
    if args.dataset == "CHB_MIT" and getattr(
        args, "chbmit_grouped_manifest", ""
    ):
        return Path(args.chbmit_grouped_manifest)
    filename = (
        "tuab_effective_split.json"
        if args.dataset == "TUAB"
        else "chbmit_effective_split.json"
    )
    return Path(args.split_manifest_dir) / filename


def _verify_read_only_binary_lock(lock_path):
    if lock_path.stat().st_mode & 0o222:
        raise ValueError(
            "final config lock is writable; recreate it with the lock tool"
        )
    ledger_path = lock_path.parent / "lock_ledger.jsonl"
    if not ledger_path.is_file():
        raise ValueError("final config lock ledger is missing")
    lock_hash = artifact_metadata(lock_path)["sha256"]
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not any(
        record.get("lock", {}).get("sha256") == lock_hash
        for record in records
    ):
        raise ValueError("final config lock hash is absent from its ledger")


def verify_binary_evaluation_lock(args, current_source_snapshot):
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
            raise ValueError(
                "smoke_test result_dir must be under results/smoke"
            )
        return {
            "mode": "smoke_test",
            "paper_evidence": False,
            "test_access_permitted": False,
        }

    if not args.final_config_lock:
        raise ValueError(
            "final_test requires --final_config_lock created from "
            "validation-only evidence"
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
    _verify_read_only_binary_lock(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("lock_version") != 2:
        raise ValueError("final config lock must use lock_version 2")
    locked_config = lock.get("locked_config")
    if not isinstance(locked_config, dict):
        raise ValueError("final config lock is missing locked_config")
    missing = [
        field
        for field in BINARY_FINAL_LOCK_FIELDS
        if field not in locked_config
    ]
    if missing:
        raise ValueError(f"final config lock is missing fields: {missing}")
    mismatches = {
        field: {
            "locked": locked_config[field],
            "requested": getattr(args, field),
        }
        for field in BINARY_FINAL_LOCK_FIELDS
        if locked_config[field] != getattr(args, field)
    }
    if mismatches:
        raise ValueError(
            "requested final-test config differs from lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not lock.get("validation_evidence"):
        raise ValueError("final config lock lacks validation_evidence")
    manifest = artifact_metadata(_binary_manifest_path(args))
    if lock.get("split_manifest_sha256") != manifest["sha256"]:
        raise ValueError(
            "binary split manifest hash differs from final config lock"
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
    threshold = lock.get("decision_threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("final config lock lacks a valid decision threshold")
    return {
        "mode": "final_test",
        "lock": artifact_metadata(lock_path),
        "split_manifest": manifest,
        "selected_checkpoint": checkpoint_record,
        "source_snapshot": locked_snapshot,
        "decision_threshold": float(threshold),
    }


def verify_binary_split_manifest(
    args,
    dataset,
    train_files,
    val_files,
    train_subjects,
    val_subjects,
    test_files=None,
):
    filename = (
        "tuab_effective_split.json"
        if dataset == "TUAB"
        else "chbmit_effective_split.json"
    )
    manifest_path = Path(args.split_manifest_dir) / filename
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen {dataset} split manifest is required: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset:
        raise ValueError(f"Unexpected dataset in {manifest_path}")
    expected = {
        "train_file_count": len(train_files),
        "val_file_count": len(val_files),
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"{dataset} manifest mismatch for {field}: "
                f"expected current value {value!r}"
            )
    inventory = manifest.get("file_inventory_sha256", {})
    for split, files in (("train", train_files), ("val", val_files)):
        current_hash = file_inventory_sha256(files)
        if inventory.get(split) != current_hash:
            raise RuntimeError(
                f"{dataset} {split} file inventory differs from manifest"
            )

    test_subjects = set(manifest.get("test_subjects", []))
    if test_files is not None:
        current_test_subjects = {
            binary_subject_id(name, dataset) for name in test_files
        }
        if manifest.get("test_file_count") != len(test_files):
            raise RuntimeError(
                f"{dataset} test file count differs from manifest"
            )
        if current_test_subjects != test_subjects:
            raise RuntimeError(
                f"{dataset} test subjects differ from manifest"
            )
        if inventory.get("test") != file_inventory_sha256(test_files):
            raise RuntimeError(
                f"{dataset} test file inventory differs from manifest"
            )
    if train_subjects & val_subjects:
        raise RuntimeError(f"{dataset} train/validation subject overlap")
    if train_subjects & test_subjects or val_subjects & test_subjects:
        raise RuntimeError(f"{dataset} test subject overlap in manifest")
    args.effective_split_manifest = str(manifest_path)
    return manifest


def load_chbmit_grouped_fold(args, include_test=False):
    """Resolve one patient-grouped CHB-MIT fold across legacy directories."""
    manifest_path = Path(args.chbmit_grouped_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_version = manifest.get("manifest_version")
    if manifest_version not in (2, 3):
        raise ValueError("unsupported CHB-MIT grouped manifest version")
    if manifest.get("dataset") != "CHB_MIT":
        raise ValueError("grouped manifest is not for CHB_MIT")
    expected_protocols = {
        2: (
            "patient-grouped cross-validation over the complete legacy "
            "directory tree"
        ),
        3: (
            "nested patient-grouped train-validation-test evaluation over "
            "the complete legacy directory tree"
        ),
    }
    if manifest.get("protocol") != expected_protocols[manifest_version]:
        raise ValueError("unexpected CHB-MIT grouped protocol")
    if include_test and manifest_version != 3:
        raise ValueError(
            "CHB-MIT final test requires a version-3 nested patient manifest"
        )
    root = Path(args.chbmit_root)
    recorded_root = Path(manifest.get("root", ""))
    if root.resolve() != recorded_root.resolve():
        raise ValueError(
            f"CHB-MIT root differs from grouped manifest: {root} vs {recorded_root}"
        )
    relative_paths = enumerate_chbmit_windows(root)
    current_hash = relative_inventory_sha256(relative_paths)
    if current_hash != manifest.get("relative_path_inventory_sha256"):
        raise RuntimeError("CHB-MIT relative-path inventory differs from manifest")
    if len(relative_paths) != manifest.get("total_window_count"):
        raise RuntimeError("CHB-MIT window count differs from grouped manifest")

    fold_index = int(args.chbmit_fold_index)
    folds = manifest.get("folds", [])
    matches = [fold for fold in folds if fold.get("fold_index") == fold_index]
    if len(matches) != 1:
        raise ValueError(f"CHB-MIT fold index {fold_index} is not unique")
    fold = matches[0]
    train_subjects = set(fold.get("train_patients", []))
    val_subjects = set(fold.get("validation_patients", []))
    test_subjects = set(fold.get("test_patients", []))
    if not train_subjects or not val_subjects:
        raise ValueError("CHB-MIT grouped fold has an empty partition")
    if manifest_version == 3 and not test_subjects:
        raise ValueError("CHB-MIT nested fold has an empty test partition")
    if (
        train_subjects & val_subjects
        or train_subjects & test_subjects
        or val_subjects & test_subjects
    ):
        raise RuntimeError("CHB-MIT canonical patient overlap in grouped fold")
    known_patients = {
        record["patient_id"] for record in manifest.get("patients", [])
    }
    if train_subjects | val_subjects | test_subjects != known_patients:
        raise RuntimeError("CHB-MIT grouped fold does not cover every patient")

    train_files = []
    val_files = []
    test_files = []
    for relative_path in relative_paths:
        patient = canonical_chbmit_patient_id(relative_path)
        if patient in train_subjects:
            train_files.append(relative_path)
        elif patient in val_subjects:
            val_files.append(relative_path)
        elif patient in test_subjects:
            test_files.append(relative_path)
        else:
            raise RuntimeError(f"unassigned CHB-MIT patient: {patient}")
    if len(train_files) != fold.get("train_window_count"):
        raise RuntimeError("CHB-MIT grouped train count differs from manifest")
    if len(val_files) != fold.get("validation_window_count"):
        raise RuntimeError("CHB-MIT grouped validation count differs from manifest")
    if manifest_version == 3 and len(test_files) != fold.get(
        "test_window_count"
    ):
        raise RuntimeError("CHB-MIT grouped test count differs from manifest")
    args.effective_split_manifest = str(manifest_path)
    args.chbmit_grouped_protocol = manifest["protocol"]
    return (
        train_files,
        val_files,
        test_files if include_test else None,
        train_subjects,
        val_subjects,
        test_subjects,
        manifest,
    )


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_probabilities = None
        self.validation_labels = None
        self.test_probabilities = None
        self.test_labels = None

    def on_train_epoch_start(self):
        if self.args.transfer_mode == "frozen_linear":
            self.model.moe.eval()

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        task_loss = BCE(prob, y)  # focal_loss(prob, y)
        routing_aux_loss = getattr(self.model, "routing_aux_loss", None)
        loss = task_loss
        if routing_aux_loss is not None:
            loss = loss + self.args.router_aux_loss_coef * routing_aux_loss
            self.log("train_router_aux_loss", routing_aux_loss)
        self.log("train_task_loss", task_loss)
        # loss = focal_loss(prob,y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def validation_epoch_end(self, val_step_outputs):
        result = np.array([])
        gt = np.array([])
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        self.validation_probabilities = result.astype(np.float32)
        self.validation_labels = gt.astype(np.int64)

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def test_epoch_end(self, test_step_outputs):
        result = np.array([])
        gt = np.array([])
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        self.test_probabilities = result.astype(np.float32)
        self.test_labels = gt.astype(np.int64)
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

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


def prepare_TUAB_dataloader(args, include_test=False):
    #['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF']
    # root = "/home/dataset/tuh_eeg/tuh_eeg_abnormal/tuh_eeg_abnormal/v3.0.1/edf/processedchannel"

    #substract
    root = args.tuab_root
    if args.allow_subject_overlap:
        raise ValueError(
            "Revised runs require the frozen subject-disjoint TUAB manifest; "
            "legacy overlap reproduction must use the archived legacy runner"
        )
    train_files = sorted(
        name
        for name in os.listdir(os.path.join(root, "train"))
        if name.endswith(".pkl")
    )
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = sorted(
        name
        for name in os.listdir(os.path.join(root, "val"))
        if name.endswith(".pkl")
    )
    test_files = None
    if include_test:
        test_files = sorted(
            name
            for name in os.listdir(os.path.join(root, "test"))
            if name.endswith(".pkl")
        )

    def subject_from_filename(filename):
        return binary_subject_id(filename, "TUAB")

    original_train_count = len(train_files)
    train_subjects = {subject_from_filename(name) for name in train_files}
    val_subjects = {subject_from_filename(name) for name in val_files}
    train_val_overlap = sorted(train_subjects & val_subjects)
    if train_val_overlap and not args.allow_subject_overlap:
        overlap_set = set(train_val_overlap)
        train_files = [
            name
            for name in train_files
            if subject_from_filename(name) not in overlap_set
        ]
        train_subjects = {subject_from_filename(name) for name in train_files}
    if train_subjects & val_subjects:
        raise RuntimeError("TUAB train/validation subject overlap remains")
    verify_binary_split_manifest(
        args,
        "TUAB",
        train_files,
        val_files,
        train_subjects,
        val_subjects,
        test_files,
    )

    print(
        len(train_files),
        len(val_files),
        None if test_files is None else len(test_files),
    )



    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,

    )
    test_loader = None
    if include_test:
        test_loader = torch.utils.data.DataLoader(
            TUABLoader(
                os.path.join(root, "test"),
                test_files,
                args.sampling_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,

    )
    print(
        len(train_loader),
        len(val_loader),
        None if test_loader is None else len(test_loader),
    )
    return train_loader, test_loader, val_loader

# class CHBMITIterableLoader(IterableDataset):
#     """
# Implementation note.
# Implementation note.
#     """
#     def __init__(self, data_dir, file_names, sampling_rate):
#         super().__init__()
#         self.data_dir = data_dir
# Implementation note.
#         self.sampling_rate = sampling_rate
        
# Implementation note.
#         self.file_paths = [os.path.join(self.data_dir, fn) for fn in self.file_names]
# Implementation note.
#         self.file_paths.sort()

#     def __iter__(self):
#         """
# Implementation note.
#         """
#         worker_info = torch.utils.data.get_worker_info()
        
#         if worker_info is None:
# Implementation note.
#             files_for_this_worker = self.file_paths
#         else:
# Implementation note.
#             worker_id = worker_info.id
#             num_workers = worker_info.num_workers
#             files_for_this_worker = self.file_paths[worker_id::num_workers]
        
#         print(f"Worker {worker_id if worker_info else 'Main'} is processing {len(files_for_this_worker)} files.")

# Implementation note.
#         for file_path in files_for_this_worker:
#             try:
#                 with open(file_path, 'rb') as f:
# Implementation note.
#                     sample = pickle.load(f)['X'].astype("float32")
                
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
#                 label = 0 
                
# Implementation note.
#                 yield torch.from_numpy(sample), torch.tensor(label)

#             except Exception as e:
# Implementation note.
#                 print(f"[WARNING] Error loading file: {file_path}, error: {e}")
#                 continue

# def prepare_CHB_MIT_dataloader(args):
#     # set random seed
#     seed = 12348
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)

#     root = "/home/dataset/CHB-MIT/clean_segments"

# Implementation note.
#     train_files = os.listdir(os.path.join(root, "train"))
#     val_files = os.listdir(os.path.join(root, "val"))
#     test_files = os.listdir(os.path.join(root, "test"))

#     print(f"Total files - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Implementation note.
#     ctx = torch.multiprocessing.get_context('spawn')

# Implementation note.
# Implementation note.
# Implementation note.
#     train_loader = torch.utils.data.DataLoader(
#         CHBMITIterableLoader(os.path.join(root, "train"), train_files, args.sampling_rate),
#         batch_size=args.batch_size,
# Implementation note.
#         num_workers=args.num_workers,
#         persistent_workers=True,
#         pin_memory=True,
# Implementation note.
#     )
    
# Implementation note.
#     test_loader = torch.utils.data.DataLoader(
#         CHBMITIterableLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         persistent_workers=True,
#         pin_memory=True,
# Implementation note.
#     )
    
#     val_loader = torch.utils.data.DataLoader(
#         CHBMITIterableLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         persistent_workers=True,
#         pin_memory=True,
# Implementation note.
#     )
    
# Implementation note.
# Implementation note.
#     print("DataLoaders created successfully. Since IterableDataset is used, len(loader) is not available.")
    
#     return train_loader, test_loader, val_loader

def prepare_CHB_MIT_dataloader(args, include_test=False):
    #substract
    root = args.chbmit_root
    grouped_manifest = getattr(args, "chbmit_grouped_manifest", "")
    if grouped_manifest:
        (
            train_files,
            val_files,
            test_files,
            train_subjects,
            val_subjects,
            _,
            _,
        ) = load_chbmit_grouped_fold(args, include_test=include_test)
        train_root = root
        val_root = root
        test_root = root if include_test else None
    else:
        train_files = sorted(
            name
            for name in os.listdir(os.path.join(root, "train"))
            if name.endswith(".pkl")
        )
        val_files = sorted(
            name
            for name in os.listdir(os.path.join(root, "val"))
            if name.endswith(".pkl")
        )
        test_files = None
        if include_test:
            test_files = sorted(
                name
                for name in os.listdir(os.path.join(root, "test"))
                if name.endswith(".pkl")
            )
        train_subjects = {
            binary_subject_id(name, "CHB_MIT") for name in train_files
        }
        val_subjects = {
            binary_subject_id(name, "CHB_MIT") for name in val_files
        }
        verify_binary_split_manifest(
            args,
            "CHB_MIT",
            train_files,
            val_files,
            train_subjects,
            val_subjects,
            test_files,
        )
        train_root = os.path.join(root, "train")
        val_root = os.path.join(root, "val")
        test_root = os.path.join(root, "test") if include_test else None

    print(
        len(train_files),
        len(val_files),
        None if test_files is None else len(test_files),
    )

    # Implementation note.
    ctx = torch.multiprocessing.get_context('spawn')

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(train_root, train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        # multiprocessing_context=ctx,
    )
    test_loader = None
    if include_test:
        test_loader = torch.utils.data.DataLoader(
            CHBMITLoader(
                test_root,
                test_files,
                args.sampling_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(val_root, val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        # multiprocessing_context=ctx,
    )
    print(
        len(train_loader),
        len(val_loader),
        None if test_loader is None else len(test_loader),
    )
    return train_loader, test_loader, val_loader

from model.dwmoespace import dwmoespace
from model.dwmoespace_stack import dwmoespace_stack
from model.dwmoespace_newgate import dwmoespace_newgate
from model.dwmoespace_sparse_topk import dwmoespace_sparse_topk
from model.dw import dw

def supervised(args):
    args.resolved_checkpoint_metric = (
        ("val_bacc" if args.dataset == "TUAB" else "val_pr_auc")
        if args.checkpoint_metric == "auto"
        else args.checkpoint_metric
    )
    source_snapshot_at_start = source_snapshot_metadata()
    seed_everything(args.seed, args.deterministic)
    pl.seed_everything(args.seed, workers=True)
    evaluation_lock = verify_binary_evaluation_lock(
        args, source_snapshot_at_start
    )
    # get data loaders
    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(
            args,
            include_test=args.evaluation_mode == "final_test",
        )
    elif args.dataset == "CHB_MIT":
        train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(
            args,
            include_test=args.evaluation_mode == "final_test",
        )
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
            # Implementation note.
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
                "Sparse Top-K screening is from scratch; pretrained loading "
                "requires a separately verified checkpoint adapter"
            )
        if args.transfer_mode != "full_finetune":
            raise ValueError(
                "Sparse Top-K screening currently supports full_finetune only"
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
        #devices=2,
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
        lightning_model.threshold = evaluation_lock["decision_threshold"]
    split_record = artifact_metadata(args.effective_split_manifest)
    diagnostics = {
        "evaluation_mode": args.evaluation_mode,
        "evaluation_lock": evaluation_lock,
        "best_checkpoint": checkpoint_record,
        "split_manifest": split_record,
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
                lightning_model.validation_probabilities,
                lightning_model.validation_labels,
                sample_ids=validation_files,
                subject_ids=[
                    binary_subject_id(filename, args.dataset)
                    for filename in validation_files
                ],
                extra_arrays={
                    "decision_threshold": np.full(
                        lightning_model.validation_labels.shape[0],
                        lightning_model.threshold,
                        dtype=np.float32,
                    )
                },
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
            lightning_model.test_probabilities,
            lightning_model.test_labels,
            sample_ids=test_files,
            subject_ids=[
                binary_subject_id(filename, args.dataset)
                for filename in test_files
            ],
            extra_arrays={
                "decision_threshold": np.full(
                    lightning_model.test_labels.shape[0],
                    lightning_model.threshold,
                    dtype=np.float32,
                )
            },
        )
        diagnostics["prediction_artifact"] = prediction_record
        diagnostics["decision_threshold_from_validation"] = float(
            lightning_model.threshold
        )
        diagnostics["last_test_batch_gate_summary"] = gate_diagnostics(
            lightning_model.model
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


#python run_moe.py --batch_size 512 --dataset TUAB --model dwmoespace --n_classes 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,default=128, help="batch size")
    parser.add_argument("--num_workers", type=int,default=8, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
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
    parser.add_argument("--top_k",type=int,default=3,help="top k expert")
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=1.0,
        help="softmax temperature used by the sparse sample router",
    )
    parser.add_argument(
        "--router_aux_loss_coef",
        type=float,
        default=0.01,
        help="coefficient for sparse-router load balancing",
    )
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
        choices=["auto", "val_bacc", "val_pr_auc", "val_auroc"],
        default="auto",
        help="auto uses balanced accuracy for TUAB and PR-AUC for CHB-MIT",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="results/gate_ablation")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument(
        "--evaluation_mode",
        choices=["validation_only", "smoke_test", "final_test"],
        default="validation_only",
    )
    parser.add_argument("--final_config_lock", type=str, default="")
    parser.add_argument("--split_manifest_dir", type=str, default="results/manifests")
    parser.add_argument(
        "--allow_subject_overlap",
        action="store_true",
        help="Reproduce the legacy TUAB split even when train/val subjects overlap.",
    )
    parser.add_argument(
        "--tuab_root",
        type=str,
        default="/home/dataset/tuh_eeg/tuh_eeg_abnormal/tuh_eeg_abnormal/v3.0.1/edf/processed",
    )
    parser.add_argument(
        "--chbmit_root",
        type=str,
        default="/home/dataset/CHB-MIT/clean_segments",
    )
    parser.add_argument(
        "--chbmit_grouped_manifest",
        type=str,
        default="",
        help=(
            "patient-grouped CHB-MIT manifest; when set, legacy directory "
            "names are source locations rather than split assignments"
        ),
    )
    parser.add_argument(
        "--chbmit_fold_index",
        type=int,
        default=-1,
        help="validation fold index in --chbmit_grouped_manifest",
    )
    args = parser.parse_args()
    print(args)

    supervised(args)

