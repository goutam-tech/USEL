import matplotlib
import numpy as np

matplotlib.use("Agg")

import pytest

from usel.dirac.visualization import (
    plot_probability,
    plot_probability_evolution,
    plot_spin_expectation,
    plot_spinor_components,
)


def test_plot_probability():

    x = np.linspace(0, 1, 10)

    probability = np.ones(10)

    plot_probability(x, probability)


def test_plot_spinor_components():

    x = np.linspace(0, 1, 10)

    spinor = np.ones((4, 10))

    plot_spinor_components(x, spinor)


def test_invalid_spinor_plot():

    x = np.linspace(0, 1, 10)

    spinor = np.ones((2, 10))

    with pytest.raises(ValueError):
        plot_spinor_components(x, spinor)


def test_plot_spin_expectation():

    t = np.arange(5)

    spin = np.ones((5, 3))

    plot_spin_expectation(t, spin)


def test_probability_evolution():

    x = np.linspace(0, 1, 10)

    history = np.ones((5, 10))

    plot_probability_evolution(x, history)
