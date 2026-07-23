"""
Time-dependent Schrödinger evolution.

Supports:
- Exact propagator evolution
- Boundary conditions
- Normalization
"""

from __future__ import annotations
import numpy as np
from .propagators import (
    exact,
    apply
)
from .normalization import normalize

def evolve(
    psi,
    hamiltonian,
    dt,
    steps,
    dx,
    boundary=None,
    hbar=1.0
):
    """
    Time evolution:

    ψ(t+dt)=Uψ(t)

    Parameters
    ----------
    psi:
        Initial wavefunction

    hamiltonian:
        Hamiltonian operator

    dt:
        Time step

    steps:
        Number of iterations

    dx:
        Grid spacing

    boundary:
        Boundary condition object
    """


    U = exact(
        hamiltonian,
        dt,
        hbar
    )


    states=[]


    current = psi.copy()

    for _ in range(steps):

        current = apply(
            U,
            current
        )

        if boundary:

            current = boundary.apply(
                current
            )

        current = normalize(
            current,
            dx
        )

        states.append(
            current.copy()
        )

    return np.array(states)