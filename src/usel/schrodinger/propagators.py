"""
Quantum propagators.

Low level evolution operators.
"""

import numpy as np
from scipy.linalg import expm

def exact(
    hamiltonian,
    dt,
    hbar=1.0
):

    return expm(
        -1j*
        hamiltonian.matrix.toarray()
        *
        dt/
        hbar
    )

def apply(
    U,
    psi
):

    return U @ psi

def unitary_error(
    U
):

    I=np.eye(
        U.shape[0]
    )

    return np.linalg.norm(
        U.conj().T@U-I
    )