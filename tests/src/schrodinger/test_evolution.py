from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from usel.schrodinger.boundary_conditions import DirichletBoundary, PeriodicBoundary
from usel.schrodinger.evolution import evolve

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)


class MockHamiltonian:
    def __init__(self, matrix: np.ndarray):
        self.matrix = csr_matrix(matrix)


def test_output_shape_matches_steps_and_dimension():
    ham = MockHamiltonian(SIGMA_X)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    states = evolve(psi0, ham, dt=0.05, steps=30, dx=1.0)
    assert states.shape == (30, 2)


def test_does_not_mutate_input_state():
    ham = MockHamiltonian(SIGMA_X)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi0_copy = psi0.copy()
    evolve(psi0, ham, dt=0.05, steps=10, dx=1.0)
    assert np.array_equal(psi0, psi0_copy)


def test_norm_is_conserved_without_boundary():
    ham = MockHamiltonian(SIGMA_X + 0.3 * np.array([[1, 0], [0, -1]], dtype=complex))
    psi0 = np.array([0.6, 0.8j], dtype=complex)
    psi0 /= np.linalg.norm(psi0)

    states = evolve(psi0, ham, dt=0.02, steps=100, dx=1.0)
    norms = np.linalg.norm(states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_matches_exact_unitary_evolution_when_no_boundary():
    """Without a boundary condition, evolve() just repeatedly applies the
    exact propagator, so state N should equal U^N psi0 exactly (up to the
    renormalization step, which is a no-op here since U is unitary)."""
    ham = MockHamiltonian(SIGMA_X)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    dt, steps = 0.01, 40

    states = evolve(psi0, ham, dt=dt, steps=steps, dx=1.0)

    U = expm(-1j * SIGMA_X * dt)
    expected = np.linalg.matrix_power(U, steps) @ psi0
    assert np.allclose(states[-1], expected, atol=1e-8)


def test_dirichlet_boundary_zeroes_endpoints_every_step():
    n = 10
    ham = MockHamiltonian(np.eye(n, dtype=complex))
    rng = np.random.default_rng(0)
    psi0 = rng.random(n) + 1j * rng.random(n)

    states = evolve(psi0, ham, dt=0.05, steps=5, dx=0.1, boundary=DirichletBoundary())

    for state in states:
        assert state[0] == 0
        assert state[-1] == 0


def test_periodic_boundary_forces_endpoint_equality():
    n = 10
    ham = MockHamiltonian(np.eye(n, dtype=complex))
    rng = np.random.default_rng(1)
    psi0 = rng.random(n) + 1j * rng.random(n)

    states = evolve(psi0, ham, dt=0.05, steps=5, dx=0.1, boundary=PeriodicBoundary())

    for state in states:
        assert state[-1] == pytest.approx(state[0])


def test_states_remain_normalized_with_boundary_condition():
    """evolve() renormalizes after applying the boundary condition, so
    every returned state should have unit norm even though Dirichlet
    zeroing is not itself norm-preserving."""
    n = 20
    ham = MockHamiltonian(np.eye(n, dtype=complex))
    rng = np.random.default_rng(2)
    psi0 = rng.random(n) + 1j * rng.random(n)
    dx = 0.2

    states = evolve(psi0, ham, dt=0.01, steps=15, dx=dx, boundary=DirichletBoundary())

    for state in states:
        norm = np.sqrt(np.sum(np.abs(state) ** 2) * dx)
        assert norm == pytest.approx(1.0, abs=1e-9)
