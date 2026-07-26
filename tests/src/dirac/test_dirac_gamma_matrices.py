import numpy as np

from usel.dirac.gamma_matrices import (
    alpha,
    beta,
    gamma_0,
    gamma_1,
    gamma_2,
    gamma_3,
    gamma_matrices,
    identity_2,
    sigma_x,
    sigma_y,
    sigma_z,
    verify_clifford_algebra,
)


def test_pauli_matrix_shapes():
    assert sigma_x().shape == (2, 2)
    assert sigma_y().shape == (2, 2)
    assert sigma_z().shape == (2, 2)


def test_pauli_matrices_are_hermitian():
    for sigma in [sigma_x(), sigma_y(), sigma_z()]:
        assert np.allclose(sigma, sigma.conj().T)


def test_identity_matrix():
    L = identity_2()

    assert np.allclose(L, np.eye(2))


def test_gamma_matrix_shapes():
    for gamma in gamma_matrices():
        assert gamma.shape == (4, 4)


def test_gamma_zero_square():
    g0 = gamma_0()

    assert np.allclose(g0 @ g0, np.eye(4))


def test_gamma_spatial_square():
    for g in [gamma_1(), gamma_2(), gamma_3()]:
        assert np.allclose(g @ g, -np.eye(4))


def test_beta_equals_gamma_zero():
    assert np.allclose(beta(), gamma_0())


def test_alpha_dimensions():
    for i in [1, 2, 3]:
        result = alpha(i)

        assert result.shape == (4, 4)


def test_clifford_algebra():
    assert verify_clifford_algebra()
