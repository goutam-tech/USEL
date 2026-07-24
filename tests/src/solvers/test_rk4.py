from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from usel.solvers.rk4 import evolve

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class MatrixHamiltonian:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def apply(self, state: np.ndarray) -> np.ndarray:
        return self.matrix @ state


def test_output_shape_matches_steps_and_dimension():
    psi0 = np.array([1.0, 0.0], dtype=complex)
    ham = MatrixHamiltonian(np.zeros((2, 2), dtype=complex))
    states = evolve(psi0, ham, dt=0.01, steps=50)
    assert states.shape == (50, 2)


def test_does_not_mutate_input_state():
    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi0_copy = psi0.copy()
    ham = MatrixHamiltonian(SIGMA_X)
    evolve(psi0, ham, dt=0.01, steps=10)
    assert np.array_equal(psi0, psi0_copy)


def test_zero_hamiltonian_leaves_state_unchanged():
    psi0 = np.array([0.6, 0.8j], dtype=complex)
    ham = MatrixHamiltonian(np.zeros((2, 2), dtype=complex))
    states = evolve(psi0, ham, dt=0.05, steps=20)
    for state in states:
        assert np.allclose(state, psi0)


def test_zero_steps_returns_empty_array():
    psi0 = np.array([1.0, 0.0], dtype=complex)
    ham = MatrixHamiltonian(SIGMA_X)
    states = evolve(psi0, ham, dt=0.01, steps=0)
    assert states.shape == (0,)


@pytest.mark.parametrize("hbar", [1.0, 2.0])
def test_two_level_rabi_oscillation_matches_analytic(hbar):
    psi0 = np.array([1.0, 0.0], dtype=complex)
    ham = MatrixHamiltonian(SIGMA_X)
    dt, steps = 1e-3, 500

    states = evolve(psi0, ham, dt=dt, steps=steps, hbar=hbar)

    t_final = dt * steps
    exact_final = expm(-1j * SIGMA_X * t_final / hbar) @ psi0

    assert np.allclose(states[-1], exact_final, atol=1e-6)


def test_two_level_sigma_z_phase_evolution():
    psi0 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    ham = MatrixHamiltonian(SIGMA_Z)
    dt, steps = 1e-3, 300

    states = evolve(psi0, ham, dt=dt, steps=steps)

    magnitudes = np.abs(states)
    assert np.allclose(magnitudes, 1 / np.sqrt(2), atol=1e-4)

    t_final = dt * steps
    exact_final = expm(-1j * SIGMA_Z * t_final) @ psi0
    assert np.allclose(states[-1], exact_final, atol=1e-6)


def test_norm_is_conserved_for_hermitian_hamiltonian():
    psi0 = np.array([0.6 + 0.1j, 0.8 - 0.05j], dtype=complex)
    psi0 /= np.linalg.norm(psi0)
    ham = MatrixHamiltonian(SIGMA_X + 0.3 * SIGMA_Z)

    states = evolve(psi0, ham, dt=1e-3, steps=1000)
    norms = np.linalg.norm(states, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-6)


def test_smaller_dt_reduces_global_error():
    psi0 = np.array([1.0, 0.0], dtype=complex)
    ham = MatrixHamiltonian(SIGMA_X)
    t_final = 1.0

    def final_error(dt):
        steps = int(round(t_final / dt))
        states = evolve(psi0, ham, dt=dt, steps=steps)
        exact = expm(-1j * SIGMA_X * t_final) @ psi0
        return np.linalg.norm(states[-1] - exact)

    err_coarse = final_error(0.05)
    err_fine = final_error(0.025)

    assert err_fine < err_coarse
