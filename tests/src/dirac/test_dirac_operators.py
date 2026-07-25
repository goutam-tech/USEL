import numpy as np

from usel.dirac.operators import DiracOperators


def test_derivative_matrix_shape():

    x = np.linspace(0, 1, 10)

    op = DiracOperators(x)

    D = op.derivative()

    assert D.shape == (10, 10)


def test_derivative_matrix_structure():

    x = np.linspace(0, 1, 5)

    op = DiracOperators(x)

    D = op.derivative()

    assert np.allclose(np.diag(D, 1), 1 / (2 * op.dx))

    assert np.allclose(np.diag(D, -1), -1 / (2 * op.dx))


def test_momentum_operator():

    x = np.linspace(0, 1, 8)

    op = DiracOperators(x)

    p = op.momentum()

    assert p.shape == (8, 8)

    assert np.iscomplexobj(p)


def test_kinetic_operator_shape():

    x = np.linspace(0, 1, 8)

    op = DiracOperators(x)

    K = op.kinetic()

    assert K.shape == (32, 32)
