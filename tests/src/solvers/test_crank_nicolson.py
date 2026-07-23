import numpy as np
from scipy.sparse import csr_matrix
from usel.solvers.crank_nicolson import evolve


class DummyHamiltonian:
    def __init__(self, size):

        self.matrix = csr_matrix(np.eye(size, dtype=complex))


def test_crank_nicolson_steps():

    psi = np.array([1 + 0j, 0 + 0j])

    result = evolve(psi, DummyHamiltonian(2), dt=0.01, steps=5)

    assert len(result) == 5


def test_crank_nicolson_state_dimension():

    psi = np.array([1 + 0j, 0 + 0j])

    result = evolve(psi, DummyHamiltonian(2), dt=0.01, steps=3)

    assert result.shape == (3, 2)


def test_crank_nicolson_complex_state():

    psi = np.array([1 + 0j, 1j])

    result = evolve(psi, DummyHamiltonian(2), dt=0.1, steps=2)

    assert np.iscomplexobj(result)
