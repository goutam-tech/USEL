"""
Crank Nicolson time evolution solver.
"""

from __future__ import annotations

import numpy as np

from scipy.sparse import eye
from scipy.sparse.linalg import spsolve



def evolve(
    psi,
    hamiltonian,
    dt,
    steps,
    hbar=1.0
):

    H = hamiltonian.matrix


    n = H.shape[0]


    I = eye(
        n,
        format="csr"
    )


    A = (
        I
        +
        1j*dt*H/(2*hbar)
    )


    B = (
        I
        -
        1j*dt*H/(2*hbar)
    )


    states=[]


    current = psi.copy()


    for _ in range(steps):

        current = spsolve(
            A,
            B @ current
        )

        states.append(
            current.copy()
        )


    return np.array(states)