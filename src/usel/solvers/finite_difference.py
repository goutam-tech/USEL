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

    energies = np.asarray(energies)
    states = np.asarray(states)

    if energies.ndim != 1:
        raise ValueError("Eigenenergies must be a one-dimensional array.")

    if states.ndim != 2 or states.shape[1] != energies.size:
        raise ValueError("Eigenstates must have one column per eigenenergy.")

    order = np.argsort(energies, kind="stable")

    return (energies[order], states[:, order])
