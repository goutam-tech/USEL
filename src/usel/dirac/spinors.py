from __future__ import annotations

import numpy as np

PAULI_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.complex128)


PAULI_Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


PAULI_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=np.complex128)


PAULI_I: np.ndarray = np.eye(2, dtype=np.complex128)


DIRAC_I: np.ndarray = np.eye(4, dtype=np.complex128)


def gamma_matrices_dirac():
    zero = np.zeros((2, 2), dtype=np.complex128)

    gamma0 = np.block([[PAULI_I, zero], [zero, -PAULI_I]])

    gamma1 = np.block([[zero, PAULI_X], [-PAULI_X, zero]])

    gamma2 = np.block([[zero, PAULI_Y], [-PAULI_Y, zero]])

    gamma3 = np.block([[zero, PAULI_Z], [-PAULI_Z, zero]])

    return (gamma0, gamma1, gamma2, gamma3)


def dirac_alpha_matrices():
    g0, g1, g2, g3 = gamma_matrices_dirac()

    return (g0 @ g1, g0 @ g2, g0 @ g3)


def dirac_beta():
    return gamma_matrices_dirac()[0]


def spin_up():
    return np.array([1, 0], dtype=np.complex128)


def spin_down():
    return np.array([0, 1], dtype=np.complex128)


def positive_energy_spinor(momentum: np.ndarray, mass: float, energy: float, spin: str = "up"):
    if spin == "up":
        chi = spin_up()

    elif spin == "down":
        chi = spin_down()

    else:
        raise ValueError("spin must be up or down")

    px, py, pz = momentum

    sigma_p = px * PAULI_X + py * PAULI_Y + pz * PAULI_Z

    lower = (sigma_p @ chi) / (energy + mass)

    factor = np.sqrt((energy + mass) / (2 * mass))

    return factor * np.concatenate([chi, lower])


def negative_energy_spinor(momentum: np.ndarray, mass: float, energy: float, spin: str = "up"):
    if spin == "up":
        chi = spin_up()

    elif spin == "down":
        chi = spin_down()

    else:
        raise ValueError("spin must be up or down")

    px, py, pz = momentum

    sigma_p = px * PAULI_X + py * PAULI_Y + pz * PAULI_Z

    upper = (sigma_p @ chi) / (energy + mass)

    return np.concatenate([upper, chi])


def normalize_spinor(spinor: np.ndarray, dx: float = 1.0):
    norm = np.sqrt(np.sum(np.abs(spinor) ** 2) * dx)

    if norm == 0:
        return spinor

    return spinor / norm


def spinor_norm(spinor: np.ndarray):
    return np.vdot(spinor, spinor)


def spin_operator_z():
    zero = np.zeros((2, 2), dtype=np.complex128)

    sigma = np.block([[PAULI_Z, zero], [zero, PAULI_Z]])

    return sigma / 2


def spin_expectation(spinor: np.ndarray, operator=None):
    if operator is None:
        operator = spin_operator_z()

    return np.real(np.vdot(spinor, operator @ spinor))


def verify_spinor_dimension(spinor):
    return spinor.shape[0] == 4
