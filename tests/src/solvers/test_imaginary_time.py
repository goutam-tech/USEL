import numpy as np
from scipy.sparse import csr_matrix

from usel.solvers.imaginary_time import ground_state


class DummyHamiltonian:
    def __init__(self):
        self.matrix = csr_matrix(np.array([[1, 0], [0, 2]], dtype=float))


def test_ground_state_shape():
    initial = np.array([1, 1], dtype=float)

    result = ground_state(DummyHamiltonian(), initial, dt=0.01, iterations=5, dx=1.0)

    assert result.shape == (2,)


def test_ground_state_normalization():
    initial = np.array([1, 1], dtype=float)

    result = ground_state(DummyHamiltonian(), initial, dt=0.01, iterations=10, dx=1.0)

    norm = np.sqrt(np.sum(abs(result) ** 2))

    assert norm == 1.0


def test_ground_state_returns_array():
    initial = np.array([1, 0], dtype=float)

    result = ground_state(DummyHamiltonian(), initial, dt=0.1, iterations=2, dx=1)

    assert isinstance(result, np.ndarray)
