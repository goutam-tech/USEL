"""
Wavefunction normalization utilities.

Quantum states must satisfy:

∫ |ψ(x)|² dx = 1
"""

from __future__ import annotations
import numpy as np

def norm(
    psi: np.ndarray,
    dx: float
) -> float:
    """
    Calculate wavefunction norm.

    Parameters
    ----------
    psi:
        Wavefunction

    dx:
        Spatial grid spacing

    Returns
    -------
    float
        Norm value
    """

    return np.sqrt(
        np.sum(
            np.abs(psi)**2
        )
        *
        dx
    )

def normalize(
    psi: np.ndarray,
    dx: float
) -> np.ndarray:
    """
    Normalize wavefunction.

    ψ = ψ / sqrt(∫|ψ|²dx)
    """

    value = norm(
        psi,
        dx
    )

    if value == 0:
        raise ValueError(
            "Cannot normalize zero wavefunction"
        )

    return psi / value

def is_normalized(
    psi: np.ndarray,
    dx: float,
    tolerance: float = 1e-10
) -> bool:
    """
    Check normalization condition.
    """

    return abs(
        norm(psi, dx)-1
    ) < tolerance


def probability_density(
    psi: np.ndarray
) -> np.ndarray:
    """
    Calculate probability density.

    P(x)=|ψ(x)|²
    """

    return np.abs(psi)**2