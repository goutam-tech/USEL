from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from usel.schrodinger.propagators import apply, exact, unitary_error

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class MockHamiltonian:
    """Minimal stand-in exposing the sparse ``.matrix`` attribute that
    ``exact`` requires (it calls ``.toarray()`` on it)."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = csr_matrix(matrix)


def test_exact_propagator_is_unitary():
    ham = MockHamiltonian(SIGMA_X)
    U = exact(ham, dt=0.1)
    assert unitary_error(U) < 1e-10


def test_exact_propagator_identity_at_zero_dt():
    ham = MockHamiltonian(SIGMA_Z)
    U = exact(ham, dt=0.0)
    assert np.allclose(U, np.eye(2), atol=1e-12)


def test_exact_propagator_composition_property():
    """U(dt1) then U(dt2) should equal U(dt1 + dt2)."""
    ham = MockHamiltonian(SIGMA_X)
    U1 = exact(ham, dt=0.05)
    U2 = exact(ham, dt=0.05)
    U_combined = exact(ham, dt=0.1)
    assert np.allclose(U2 @ U1, U_combined, atol=1e-10)


def test_exact_propagator_scales_with_hbar():
    """Doubling hbar with the same dt should be equivalent to halving dt
    at the original hbar, since the phase depends on dt/hbar."""
    ham = MockHamiltonian(SIGMA_X)
    U_a = exact(ham, dt=0.2, hbar=2.0)
    U_b = exact(ham, dt=0.1, hbar=1.0)
    assert np.allclose(U_a, U_b, atol=1e-10)


def test_exact_matches_known_rabi_rotation():
    """H = sigma_x  =>  U(t) = cos(t) I - i sin(t) sigma_x."""
    ham = MockHamiltonian(SIGMA_X)
    t = 0.37
    U = exact(ham, dt=t)
    expected = np.cos(t) * np.eye(2) - 1j * np.sin(t) * SIGMA_X
    assert np.allclose(U, expected, atol=1e-10)


def test_apply_is_matrix_vector_product():
    U = np.array([[0, 1], [1, 0]], dtype=complex)
    psi = np.array([1.0, 0.0], dtype=complex)
    result = apply(U, psi)
    assert np.allclose(result, [0.0, 1.0])


def test_apply_preserves_norm_for_unitary_matrix():
    ham = MockHamiltonian(SIGMA_X + 0.5 * SIGMA_Z)
    U = exact(ham, dt=0.3)
    psi0 = np.array([0.6, 0.8j], dtype=complex)
    psi0 /= np.linalg.norm(psi0)
    psi1 = apply(U, psi0)
    assert np.linalg.norm(psi1) == pytest.approx(1.0, abs=1e-10)


def test_unitary_error_is_near_zero_for_identity():
    assert unitary_error(np.eye(4)) < 1e-12


def test_unitary_error_detects_non_unitary_matrix():
    non_unitary = np.array([[1.0, 0.5], [0.0, 1.0]])
    assert unitary_error(non_unitary) > 0.5


def test_unitary_error_zero_for_pauli_matrices():
    assert unitary_error(SIGMA_X) < 1e-12
    assert unitary_error(SIGMA_Z) < 1e-12
