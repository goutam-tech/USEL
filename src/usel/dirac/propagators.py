from __future__ import annotations

from scipy.linalg import expm


def unitary_propagator(H, dt, hbar=1.0):

    return expm(-1j * H * dt / hbar)


def apply_propagator(U, psi):

    n = psi.shape[1]

    state = psi.reshape(4 * n)

    state = U @ state

    return state.reshape(4, n)
