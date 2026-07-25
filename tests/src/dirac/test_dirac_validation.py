import numpy as np

from usel.dirac.validation import (
    check_normalization,
    check_probability_conservation,
    norm,
    probability_density,
)


def test_norm():
    psi = np.ones((4, 10))

    value = norm(psi)

    assert value == 40


def test_check_normalization_true():
    psi = np.ones((10,))

    psi = psi / np.sqrt(10)

    assert check_normalization(psi)


def test_check_normalization_false():
    psi = np.ones((10,))

    assert not check_normalization(psi)


def test_probability_density():
    psi = np.ones((4, 5), dtype=complex)

    rho = probability_density(psi)

    assert rho.shape == (5,)

    assert np.all(rho == 4)


def test_probability_conservation():
    assert check_probability_conservation(1.0, 1.0)

    assert not check_probability_conservation(1.0, 2.0)
