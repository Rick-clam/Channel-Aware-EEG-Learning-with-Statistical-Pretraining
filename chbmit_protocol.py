"""Patient-identity and inventory helpers for leakage-safe CHB-MIT folds."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable


CASE_TO_CANONICAL_PATIENT = {"chb21": "chb01"}
SOURCE_SPLITS = ("train", "val", "test")


def chbmit_case_id(path_or_name: str) -> str:
    """Parse a CHB-MIT case ID such as ``chb21`` from a window filename."""
    name = os.path.basename(path_or_name)
    case_id = name.split("_", 1)[0].lower()
    if not case_id.startswith("chb") or not case_id[3:].isdigit():
        raise ValueError(f"Cannot parse CHB-MIT case ID from {path_or_name!r}")
    return case_id


def canonical_chbmit_patient_id(path_or_name: str) -> str:
    """Map recording cases to canonical patients before any split operation."""
    case_id = chbmit_case_id(path_or_name)
    return CASE_TO_CANONICAL_PATIENT.get(case_id, case_id)


def enumerate_chbmit_windows(root: Path) -> list[str]:
    """Return sorted ``source_split/filename.pkl`` paths from the legacy tree."""
    root = Path(root)
    records = []
    for source_split in SOURCE_SPLITS:
        directory = root / source_split
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        records.extend(
            f"{source_split}/{entry.name}"
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.endswith(".pkl")
        )
    return sorted(records)


def relative_inventory_sha256(relative_paths: Iterable[str]) -> str:
    """Hash the complete sorted relative-path inventory."""
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        digest.update(str(relative_path).replace("\\", "/").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
