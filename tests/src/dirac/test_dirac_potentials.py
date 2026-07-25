import numpy as np

from usel.dirac.potentials import (
    ScalarPotential,
    barrier_potential,
    free_potential,
    harmonic_potential,
)


def test_free_potential():
    x = np.linspace(-5, 5, 20)

    V = free_potential(x)

    assert V.shape == x.shape

    assert np.all(V == 0)


def test_harmonic_potential():
    x = np.array([-2, 0, 2])

    V = harmonic_potential(x, strength=1)

    expected = np.array([2, 0, 2])

    assert np.allclose(V, expected)


def test_barrier_potential():
    x = np.linspace(0, 10, 11)

    V = barrier_potential(x, height=5, start=3, end=7)

    assert V[0] == 0
    assert V[5] == 5
    assert V[-1] == 0


def test_scalar_potential():
    values = np.array([1, 2, 3])

    potential = ScalarPotential(values)

    result = potential(np.array([0, 1, 2]))

    assert np.allclose(result, values)
