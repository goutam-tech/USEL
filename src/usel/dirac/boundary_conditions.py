"""
Boundary conditions for Dirac equation.

Supports:

- Dirichlet boundary
- Periodic boundary
- Absorbing boundary
"""

from __future__ import annotations

import numpy as np


def dirichlet(psi: np.ndarray):
    """
    ψ=0 at boundaries
    """

    psi[:, 0] = 0

    psi[:, -1] = 0

    return psi


def periodic(psi: np.ndarray):
    """
    Periodic boundaries.

    ψ(0)=ψ(L)
    """

    psi[:, 0] = psi[:, -1]

    return psi


def absorbing(psi: np.ndarray, strength=0.1):
    """
    Absorbing edges.

    Prevents reflection.
    """

    n = psi.shape[1]

    mask = np.ones(n)

    edge = int(n * 0.1)

    x = np.linspace(0, 1, edge)

    damping = np.sin(np.pi * x / 2) ** strength

    mask[:edge] = damping

    mask[-edge:] = damping[::-1]

    psi *= mask

    return psi
