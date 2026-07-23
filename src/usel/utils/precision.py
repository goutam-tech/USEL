"""Precision control utilities for numerical output formatting."""

from __future__ import annotations

_GLOBAL_PRECISION = 8


def set_global_precision(digits: int) -> None:
    """Set the global default rounding precision (number of decimal digits)."""
    global _GLOBAL_PRECISION
    if digits < 0:
        raise ValueError("Precision digits must be non-negative")
    _GLOBAL_PRECISION = digits


def get_global_precision() -> int:
    """Return the currently configured global rounding precision."""
    return _GLOBAL_PRECISION


def round_to_precision(value: float, digits: int | None = None) -> float:
    """Round ``value`` to ``digits`` decimal places (global precision by default)."""
    precision = digits if digits is not None else _GLOBAL_PRECISION
    return round(value, precision)
