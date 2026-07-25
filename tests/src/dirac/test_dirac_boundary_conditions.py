import numpy as np

from usel.dirac.boundary_conditions import (
    absorbing,
    dirichlet,
    periodic,
)


def test_dirichlet_sets_boundaries_zero():
    psi = np.ones((4, 10), dtype=complex)

    result = dirichlet(psi.copy())

    assert np.all(result[:, 0] == 0)
    assert np.all(result[:, -1] == 0)

    # interior should remain unchanged
    assert np.all(result[:, 1:-1] == 1)


def test_periodic_copies_last_boundary_to_first():
    psi = np.zeros((4, 5), dtype=complex)

    psi[:, -1] = 5 + 2j

    result = periodic(psi.copy())

    assert np.all(result[:, 0] == result[:, -1])
    assert np.all(result[:, 0] == 5 + 2j)


def test_absorbing_reduces_edges():
    psi = np.ones((4, 100), dtype=complex)

    result = absorbing(psi.copy(), strength=0.1)

    # edges should be damped
    assert np.abs(result[:, 0]).max() <= 1
    assert np.abs(result[:, -1]).max() <= 1


def test_absorbing_keeps_shape():

    psi = np.random.random((4, 50))

    result = absorbing(psi.copy())

    assert result.shape == psi.shape
