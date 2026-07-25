"""
Dirac gamma matrices.

Implements:
- Pauli matrices
- Gamma matrices
- Alpha matrices
- Beta matrix
- Clifford algebra validation

"""

from __future__ import annotations

import numpy as np


def sigma_x() -> np.ndarray:
    """
    Pauli X matrix.
    """

    return np.array([[0, 1], [1, 0]], dtype=complex)


def sigma_y() -> np.ndarray:
    """
    Pauli Y matrix.
    """

    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def sigma_z() -> np.ndarray:
    """
    Pauli Z matrix.
    """

    return np.array([[1, 0], [0, -1]], dtype=complex)


def identity_2():
    return np.eye(2, dtype=complex)


def gamma_0() -> np.ndarray:
    """
    Gamma zero matrix.

    γ0 =
    | I   0 |
    | 0  -I |

    """

    L = identity_2()

    return np.block([[L, np.zeros((2, 2))], [np.zeros((2, 2)), -L]])


def gamma_i(sigma: np.ndarray) -> np.ndarray:
    """
    Spatial gamma matrix.

    γi =
    | 0      σi |
    | -σi     0 |

    """

    zero = np.zeros((2, 2), dtype=complex)

    return np.block([[zero, sigma], [-sigma, zero]])


def gamma_1():
    return gamma_i(sigma_x())


def gamma_2():
    return gamma_i(sigma_y())


def gamma_3():
    return gamma_i(sigma_z())


def gamma_matrices():
    """
    Return all gamma matrices.

    γμ = (γ0,γ1,γ2,γ3)
    """

    return [gamma_0(), gamma_1(), gamma_2(), gamma_3()]


def beta():
    """
    β = γ0
    """

    return gamma_0()


def alpha(index: int):
    """
    αi = γ0γi

    index:
        1,2,3
    """

    g0 = gamma_0()

    gi = gamma_matrices()[index]

    return g0 @ gi


def verify_clifford_algebra(tolerance=1e-10):
    """
    Check:

    γμγν + γνγμ
    =
    2gμνI

    """

    gamma = gamma_matrices()

    metric = np.diag([1, -1, -1, -1])

    M = np.eye(4, dtype=complex)

    for mu in range(4):
        for nu in range(4):
            lhs = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]

            rhs = 2 * metric[mu, nu] * M

            if not np.allclose(lhs, rhs, atol=tolerance):
                return False

    return True
