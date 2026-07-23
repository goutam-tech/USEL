"""Standard quantum-mechanical potential energy functions.

Each function accepts a spatial grid ``x`` and optional parameters, returning
a 1-D ``numpy.ndarray`` of potential values on that grid.
"""

from __future__ import annotations
import numpy as np

def infinite_square_well(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Infinite square well: V = 0 inside [x_min, x_max], infinity outside.

    The returned array is 0 inside the well and ``np.inf`` outside.
    """
    v = np.where((x >= x_min) & (x <= x_max), 0.0, np.inf)
    return v

def finite_square_well(
    x: np.ndarray, x_min: float, x_max: float, depth: float
) -> np.ndarray:
    """Finite square well: V = -depth inside [x_min, x_max], 0 outside."""
    return np.where((x >= x_min) & (x <= x_max), -depth, 0.0)

def harmonic_oscillator(
    x: np.ndarray, omega: float = 1.0, mass: float = 1.0
) -> np.ndarray:
    """Quantum harmonic oscillator: V = 0.5 * mass * omega^2 * x^2."""
    return 0.5 * mass * omega**2 * x**2

def double_well(
    x: np.ndarray, barrier_height: float = 1.0, separation: float = 1.0
) -> np.ndarray:
    """Symmetric double-well potential.

    Two wells of depth ``barrier_height`` centred at ``+/- separation``.
    """
    left = -0.5 * barrier_height * np.exp(-((x + separation) ** 2) / 0.5)
    right = -0.5 * barrier_height * np.exp(-((x - separation) ** 2) / 0.5)
    return left + right

def barrier(
    x: np.ndarray, x_min: float, x_max: float, height: float
) -> np.ndarray:
    """Rectangular potential barrier: V = height in [x_min, x_max], 0 outside."""
    return np.where((x >= x_min) & (x <= x_max), height, 0.0)

def coulomb(x: np.ndarray, charge: float = -1.0) -> np.ndarray:
    """Coulomb potential: V = charge / |x|, softened at the origin.

    A small epsilon is added to avoid the singularity.
    """
    epsilon = 1e-12
    return charge / (np.abs(x) + epsilon)

def kronig_penney(
    x: np.ndarray, barrier_height: float, period: float, width: float
) -> np.ndarray:
    """Kronig-Penney periodic potential.

    Alternates between 0 and ``barrier_height`` with the given period and
    barrier width.
    """
    v = np.zeros_like(x)
    for xi in x:
        idx = int(np.floor(xi / period)) if period > 0 else 0
        pos = xi - idx * period
        if 0 <= pos < width:
            v[int(np.round((xi - x[0]) / (x[1] - x[0])))] = barrier_height
    return v