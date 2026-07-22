"""Readers for CSV, JSON, NumPy binary (.npy/.npz), and plain text files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from usel.exceptions import IOError_


def read_csv(filepath: str | Path, delimiter: str = ",", skip_header: int = 0) -> np.ndarray:
    """Read a CSV file into a NumPy array."""
    path = Path(filepath)
    if not path.exists():
        raise IOError_(f"CSV file not found: {path}")
    try:
        return np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header, dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to read CSV file {path}: {exc}") from exc


def read_json(filepath: str | Path) -> Any:
    """Read a JSON file and return the parsed object."""
    path = Path(filepath)
    if not path.exists():
        raise IOError_(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to read JSON file {path}: {exc}") from exc


def read_npy(filepath: str | Path) -> np.ndarray:
    """Read a NumPy binary (.npy) file into an array."""
    path = Path(filepath)
    if not path.exists():
        raise IOError_(f"NumPy binary file not found: {path}")
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to read .npy file {path}: {exc}") from exc


def read_text(filepath: str | Path, encoding: str = "utf-8") -> str:
    """Read a plain text file and return its contents as a string."""
    path = Path(filepath)
    if not path.exists():
        raise IOError_(f"Text file not found: {path}")
    try:
        return path.read_text(encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to read text file {path}: {exc}") from exc
