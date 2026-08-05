"""Shared utilities for reproducible CAER experiments."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import random
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and data-loader workers."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_snapshot() -> Dict[str, Any]:
    """Hash executable source files, including relevant untracked files."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        listed = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_dirty": None,
            "source_snapshot_sha256": None,
            "source_file_count": None,
        }
    suffixes = {".py", ".sh", ".yaml", ".yml", ".toml"}
    source_files = sorted(
        path
        for path in listed
        if Path(path).suffix.lower() in suffixes
        and not path.startswith(
            (
                "results/",
                "logs/",
                "log/",
                "log-pretrain/",
                "lightning_logs/",
                "wandb/",
            )
        )
    )
    digest = hashlib.sha256()
    retained = []
    for relative_path in source_files:
        path = Path(relative_path)
        if not path.is_file():
            continue
        payload = path.read_bytes()
        retained.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "git_dirty": bool(status.strip()),
        "source_snapshot_sha256": digest.hexdigest(),
        "source_file_count": len(retained),
        "source_files": retained,
    }


def source_snapshot_metadata() -> Dict[str, Any]:
    """Return the executable-source snapshot used by lock verification."""
    return _source_snapshot()


def artifact_metadata(path) -> Dict[str, Any]:
    """Return a path and SHA-256 for an immutable run artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_experiment_result(
    args: Any,
    metrics: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one auditable JSON result and append the record to JSONL."""
    output_dir = Path(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = getattr(args, "dataset", getattr(args, "logdataset", "unknown"))
    model = getattr(args, "model", "unsupervised_pretrain")
    run_id = args.run_id or (
        f"{dataset}_{model}_{getattr(args, 'gate_type', 'na')}"
        f"_seed{args.seed}"
    )
    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "metrics": _jsonable(metrics),
        "config": _jsonable(vars(args)),
        "provenance": {
            "git_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            **_source_snapshot(),
        },
    }
    if extra:
        record["diagnostics"] = _jsonable(extra)

    json_path = output_dir / f"{run_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "experiment_results.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return json_path


def save_prediction_artifact(
    args: Any,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    sample_ids=None,
    subject_ids=None,
    extra_arrays: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save held-out logits and resampling identifiers in a compressed NPZ."""
    output_dir = Path(args.result_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = getattr(args, "dataset", getattr(args, "logdataset", "unknown"))
    model = getattr(args, "model", "unknown")
    run_id = args.run_id or (
        f"{dataset}_{model}_{getattr(args, 'gate_type', 'na')}"
        f"_seed{args.seed}"
    )
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("prediction logits and labels have different lengths")
    count = labels.shape[0]
    if sample_ids is None:
        sample_ids = [str(index) for index in range(count)]
    if subject_ids is None:
        subject_ids = sample_ids
    sample_ids = np.asarray(list(sample_ids)[:count], dtype=np.str_)
    subject_ids = np.asarray(list(subject_ids)[:count], dtype=np.str_)
    if sample_ids.shape[0] != count or subject_ids.shape[0] != count:
        raise ValueError("prediction metadata is shorter than prediction arrays")

    artifact_path = output_dir / f"{run_id}.npz"
    arrays = {
        "logits": logits.astype(np.float32, copy=False),
        "labels": labels.astype(np.int64, copy=False),
        "sample_ids": sample_ids,
        "subject_ids": subject_ids,
    }
    for name, values in (extra_arrays or {}).items():
        values = np.asarray(values)
        if values.shape[0] != count:
            raise ValueError(
                f"extra prediction array {name!r} has length "
                f"{values.shape[0]}, expected {count}"
            )
        arrays[name] = values
    np.savez_compressed(artifact_path, **arrays)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {
        "path": str(artifact_path),
        "sha256": digest,
        "num_samples": int(count),
        "num_subjects": int(np.unique(subject_ids).shape[0]),
    }


def gate_diagnostics(model: torch.nn.Module) -> Dict[str, Any]:
    """Return lightweight diagnostics from the active expert gate."""
    moe = getattr(model, "moe", None)
    weights = getattr(moe, "last_gate_weights", None)
    if weights is None:
        return {}
    weights = weights.float()
    safe_weights = weights.clamp_min(1e-12)
    entropy = -(safe_weights * safe_weights.log()).sum(-1)
    return {
        "gate_mean_by_expert": weights.mean(dim=(0, 1)),
        "gate_std_by_expert": weights.std(dim=(0, 1), unbiased=False),
        "gate_entropy_mean": entropy.mean(),
        "gate_entropy_std": entropy.std(unbiased=False),
    }
