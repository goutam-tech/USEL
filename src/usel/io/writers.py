"""Writers for CSV, JSON, NumPy binary (.npy), and plain text files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from usel.exceptions import IOError_


def write_csv(data: np.ndarray, filepath: str | Path, delimiter: str = ",") -> None:
    """Write a NumPy array to a CSV file."""
    path = Path(filepath)
    try:
        np.savetxt(path, np.asarray(data), delimiter=delimiter)
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to write CSV file {path}: {exc}") from exc


def write_json(data: Any, filepath: str | Path, indent: int = 2) -> None:
    """Write a JSON-serializable object to a file."""
    path = Path(filepath)
    try:
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to write JSON file {path}: {exc}") from exc


def write_npy(data: np.ndarray, filepath: str | Path) -> None:
    """Write a NumPy array to a .npy binary file."""
    path = Path(filepath)
    try:
        np.save(path, np.asarray(data))
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to write .npy file {path}: {exc}") from exc


def write_text(content: str, filepath: str | Path, encoding: str = "utf-8") -> None:
    """Write a string to a plain text file."""
    path = Path(filepath)
    try:
        path.write_text(content, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise IOError_(f"Failed to write text file {path}: {exc}") from exc
