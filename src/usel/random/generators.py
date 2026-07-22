"""Reproducible random number, vector, matrix, and state generation.

All functions accept an optional ``seed`` for reproducibility and are built
on top of NumPy's modern ``Generator`` API (``numpy.random.default_rng``).
"""

from __future__ import annotations

import numpy as np

from usel.math.matrix import Matrix


def random_matrix(rows: int, cols: int, seed: int | None = None) -> Matrix:
    """Generate a matrix of uniform random values in [0, 1)."""
    rng = np.random.default_rng(seed)
    return Matrix(rng.random((rows, cols)))


def gaussian_numbers(
    size: int, mean: float = 0.0, std: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """Generate an array of Gaussian (normal) distributed random numbers."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=size)


def uniform_numbers(
    size: int, low: float = 0.0, high: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """Generate an array of uniformly distributed random numbers in [low, high)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=size)


def random_vector(size: int, seed: int | None = None) -> np.ndarray:
    """Generate a random vector of uniform values in [0, 1)."""
    rng = np.random.default_rng(seed)
    return rng.random(size)


def random_state(seed: int | None = None) -> np.random.Generator:
    """Return a seeded NumPy random Generator for reproducible sequences."""
    return np.random.default_rng(seed)
