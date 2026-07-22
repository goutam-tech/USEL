"""Custom exception hierarchy for usel."""

from __future__ import annotations


class uselError(Exception):
    """Base exception for all usel errors."""


class MatrixError(uselError):
    """Raised for invalid matrix operations (shape mismatch, singularity, etc.)."""


class SolverError(uselError):
    """Raised when a numerical solver fails to converge or receives bad input."""


class ConfigError(uselError):
    """Raised for configuration loading, parsing, or validation failures."""


class ValidationError(uselError):
    """Raised when input validation fails."""


class IOError_(uselError):
    """Raised for file input/output failures.

    Named ``IOError_`` to avoid shadowing the Python builtin ``IOError``.
    """


class PhysicsError(uselError):
    """Raised when a physics simulation encounters invalid parameters or diverges."""


__all__ = [
    "uselError",
    "MatrixError",
    "SolverError",
    "ConfigError",
    "ValidationError",
    "IOError_",
    "PhysicsError",
]
