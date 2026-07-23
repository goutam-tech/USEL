"""Input validation helpers used throughout usel."""

from __future__ import annotations

from typing import Any

from usel.exceptions import ValidationError


def validate_positive(value: float, name: str = "value") -> None:
    """Raise ``ValidationError`` if ``value`` is not strictly positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_range(value: float, low: float, high: float, name: str = "value") -> None:
    """Raise ``ValidationError`` if ``value`` is outside ``[low, high]``."""
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be within [{low}, {high}], got {value}")


def validate_shape(shape: tuple[int, ...], expected: tuple[int, ...], name: str = "array") -> None:
    """Raise ``ValidationError`` if ``shape`` does not equal ``expected``."""
    if shape != expected:
        raise ValidationError(f"{name} shape mismatch: expected {expected}, got {shape}")


def validate_type(value: Any, expected_type: type | tuple[type, ...], name: str = "value") -> None:
    """Raise ``ValidationError`` if ``value`` is not an instance of ``expected_type``."""
    if not isinstance(value, expected_type):
        raise ValidationError(f"{name} must be of type {expected_type}, got {type(value).__name__}")
