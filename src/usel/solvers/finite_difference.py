"""
Finite difference Schrödinger solver.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from usel.schrodinger.hamiltonian import build_hamiltonian


def solve(grid, potential, levels=5, mass=1.0, hbar=1.0):
    """
    Solve stationary Schrödinger equation.

    Returns
    -------
    energies
    states
    """

    H = build_hamiltonian(grid, potential, mass, hbar)

    energies, states = eigsh(H.matrix, k=levels, which="SA")

    order = np.argsort(energies)

    return (energies[order], states[:, order])
