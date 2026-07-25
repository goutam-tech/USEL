import numpy as np

from usel.dirac.propagators import (
    apply_propagator,
    unitary_propagator,
)


def test_unitary_propagator_identity():

    H = np.zeros((4, 4), dtype=complex)

    U = unitary_propagator(H, dt=1)

    assert np.allclose(U, np.eye(4))


def test_unitary_propagator_shape():

    H = np.eye(8, dtype=complex)

    U = unitary_propagator(H, dt=0.1)

    assert U.shape == (8, 8)


def test_apply_propagator_shape():

    U = np.eye(20, dtype=complex)

    psi = np.ones((4, 5), dtype=complex)

    result = apply_propagator(U, psi)

    assert result.shape == (4, 5)


def test_apply_identity_propagator():

    U = np.eye(12, dtype=complex)

    psi = np.random.random((4, 3))

    result = apply_propagator(U, psi)

    assert np.allclose(result, psi)
