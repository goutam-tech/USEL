from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
import matplotlib.pyplot as plt

from usel.schrodinger.visualization import (
    animate_evolution,
    plot_probability,
    plot_wavefunction,
)


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def test_plot_wavefunction_draws_real_and_imaginary_lines():
    x = np.linspace(-5, 5, 100)
    psi = np.exp(-(x**2)) * np.exp(1j * x)

    plot_wavefunction(x, psi)

    ax = plt.gca()
    assert len(ax.lines) == 2
    assert np.allclose(ax.lines[0].get_ydata(), psi.real)
    assert np.allclose(ax.lines[1].get_ydata(), psi.imag)


def test_plot_wavefunction_sets_labels_and_title():
    x = np.linspace(-5, 5, 50)
    psi = np.exp(-(x**2)).astype(complex)

    plot_wavefunction(x, psi, title="My Custom Title")

    ax = plt.gca()
    assert ax.get_xlabel() == "Position"
    assert ax.get_title() == "My Custom Title"


def test_plot_wavefunction_default_title():
    x = np.linspace(-5, 5, 50)
    psi = np.exp(-(x**2)).astype(complex)
    plot_wavefunction(x, psi)
    assert plt.gca().get_title() == "Wavefunction"


def test_plot_wavefunction_includes_legend():
    x = np.linspace(-5, 5, 50)
    psi = np.exp(-(x**2)).astype(complex)
    plot_wavefunction(x, psi)
    legend = plt.gca().get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert "Real" in labels
    assert "Imaginary" in labels


def test_plot_probability_draws_density_curve():
    x = np.linspace(-5, 5, 100)
    psi = np.exp(-(x**2)) * np.exp(1j * x)

    plot_probability(x, psi)

    ax = plt.gca()
    assert len(ax.lines) == 1
    assert np.allclose(ax.lines[0].get_ydata(), np.abs(psi) ** 2)
    assert ax.get_ylabel() == "|\u03c8|\u00b2"
    assert ax.get_title() == "Probability Density"


def test_animate_evolution_runs_without_error():
    x = np.linspace(-5, 5, 50)
    base = np.exp(-(x**2)).astype(complex)
    states = np.array([base * np.exp(-0.1j * i) for i in range(5)])

    animate_evolution(x, states)


def test_animate_evolution_initial_frame_matches_first_state():
    x = np.linspace(-5, 5, 50)
    base = np.exp(-(x**2)).astype(complex)
    states = np.array([base * (i + 1) for i in range(3)])

    animate_evolution(x, states)

    ax = plt.gca()
    assert np.allclose(ax.lines[0].get_ydata(), np.abs(states[0]) ** 2)
