import numpy as np
import pytest

from usel.dirac.spinors import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    dirac_alpha_matrices,
    dirac_beta,
    gamma_matrices_dirac,
    negative_energy_spinor,
    normalize_spinor,
    positive_energy_spinor,
    spin_down,
    spin_up,
    spinor_norm,
    verify_spinor_dimension,
)


def test_pauli_matrices_shape():
    assert PAULI_X.shape == (2, 2)
    assert PAULI_Y.shape == (2, 2)
    assert PAULI_Z.shape == (2, 2)


def test_gamma_matrices_shape():
    gammas = gamma_matrices_dirac()

    assert len(gammas) == 4

    for g in gammas:
        assert g.shape == (4, 4)


def test_alpha_matrices_shape():
    alpha = dirac_alpha_matrices()

    assert len(alpha) == 3

    for a in alpha:
        assert a.shape == (4, 4)


def test_beta_shape():
    beta = dirac_beta()

    assert beta.shape == (4, 4)


def test_spin_states():
    assert np.allclose(spin_up(), [1, 0])

    assert np.allclose(spin_down(), [0, 1])


def test_positive_energy_spinor():
    psi = positive_energy_spinor(np.array([1, 0, 0]), mass=1, energy=2)

    assert psi.shape == (4,)


def test_negative_energy_spinor():
    psi = negative_energy_spinor(np.array([1, 0, 0]), mass=1, energy=2)

    assert psi.shape == (4,)


def test_invalid_spin():
    with pytest.raises(ValueError):
        positive_energy_spinor(np.array([0, 0, 0]), 1, 1, spin="invalid")


def test_normalize_spinor():
    psi = np.array([3, 4], dtype=complex)

    result = normalize_spinor(psi)

    assert np.isclose(np.sum(np.abs(result) ** 2), 1)


def test_spinor_norm():
    psi = np.array([1, 1j])

    assert spinor_norm(psi) == 2


def test_verify_spinor_dimension():
    psi = np.zeros((4, 10))

    assert verify_spinor_dimension(psi)
