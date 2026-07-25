import numpy as np

from usel.dirac.evolution import DiracEvolution


def create_zero_hamiltonian(size):
    return np.zeros((size, size), dtype=complex)


def test_propagator_identity_for_zero_hamiltonian():
    H = create_zero_hamiltonian(8)

    evolution = DiracEvolution(H)

    U = evolution.propagator(dt=1.0)

    assert np.allclose(U, np.eye(8))


def test_propagator_shape():
    H = np.eye(8, dtype=complex)

    evolution = DiracEvolution(H)

    U = evolution.propagator(dt=0.1)

    assert U.shape == (8, 8)


def test_step_preserves_spinor_shape():
    n = 5

    H = create_zero_hamiltonian(4 * n)

    evolution = DiracEvolution(H)

    psi = np.ones((4, n), dtype=complex)

    result = evolution.step(psi, dt=0.1)

    assert result.shape == (4, n)


def test_step_applies_dirichlet_boundary():
    n = 10

    H = create_zero_hamiltonian(4 * n)

    evolution = DiracEvolution(H)

    psi = np.ones((4, n), dtype=complex)

    result = evolution.step(psi, dt=0.1)

    assert np.all(result[:, 0] == 0)
    assert np.all(result[:, -1] == 0)


def test_evolve_history_length():
    n = 5

    H = create_zero_hamiltonian(4 * n)

    evolution = DiracEvolution(H)

    psi = np.ones((4, n), dtype=complex)

    history = evolution.evolve(psi, dt=0.1, steps=10)

    assert history.shape == (11, 4, n)
