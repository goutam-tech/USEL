"""
Quantum mechanical observables.

Provides expectation values and
state comparison utilities.
"""

from __future__ import annotations
import numpy as np
from .operators import QuantumOperator

def expectation(
    psi: np.ndarray,
    operator: QuantumOperator,
    dx: float
) -> complex:
    """
    Calculate:

    <A> = ∫ ψ* A ψ dx
    """

    result = np.conjugate(
        psi
    ) * operator.apply(
        psi
    )

    return np.sum(
        result
    ) * dx

def variance(
    psi: np.ndarray,
    operator: QuantumOperator,
    dx: float
) -> float:
    """
    Calculate variance:

    <A²> - <A>²
    """

    Apsi = operator.apply(
        psi
    )

    A2psi = operator.apply(
        Apsi
    )

    mean = expectation(
        psi,
        operator,
        dx
    )

    mean_square = (
        np.sum(
            np.conjugate(psi)
            *
            A2psi
        )
        *
        dx
    )

    return float(
        np.real(
            mean_square
            -
            mean**2
        )
    )

def uncertainty(
    psi,
    operator,
    dx
):
    """
    Calculate uncertainty:

    ΔA = sqrt(variance)
    """

    return np.sqrt(
        variance(
            psi,
            operator,
            dx
        )
    )



def overlap(
    psi1: np.ndarray,
    psi2: np.ndarray,
    dx: float
) -> complex:
    """
    State overlap:

    <ψ1|ψ2>
    """

    return (
        np.sum(
            np.conjugate(psi1)
            *
            psi2
        )
        *
        dx
    )

def fidelity(
    psi1: np.ndarray,
    psi2: np.ndarray,
    dx: float
) -> float:
    """
    Quantum state fidelity.

    F = |<ψ1|ψ2>|²
    """

    return float(
        abs(
            overlap(
                psi1,
                psi2,
                dx
            )
        )
        **2
    )

def probability(
    psi: np.ndarray,
    region=None
):
    """
    Probability of finding particle.

    """

    density = (
        np.abs(psi)**2
    )

    if region is None:
        return density


    return np.sum(
        density[region]
    )