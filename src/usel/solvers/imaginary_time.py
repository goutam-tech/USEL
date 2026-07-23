"""
Imaginary time evolution solver.

Finds ground state.
"""

import numpy as np
from scipy.sparse.linalg import expm_multiply


def ground_state(hamiltonian, initial_state, dt, iterations, dx):

    H = hamiltonian.matrix

    psi = initial_state.copy()

    for _ in range(iterations):
        psi = expm_multiply(-H * dt, psi)

        norm = np.sqrt(np.sum(abs(psi) ** 2) * dx)

        psi /= norm

    return psi
