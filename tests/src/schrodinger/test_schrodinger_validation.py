from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.validation import (
    check_probability_conservation,
    energy_conservation,
    max_probability_error,
)


def _normalized_states(n_states=5, n_grid=100, dx=0.1):
    x = np.linspace(-5, 5, n_grid)
    states = []
    for i in range(n_states):
        psi = np.exp(-((x - 0.1 * i) ** 2))
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        states.append(psi.astype(complex))
    return states


def test_probability_conservation_true_for_normalized_states():
    states = _normalized_states()
    assert check_probability_conservation(states, dx=0.1)


def test_probability_conservation_false_for_unnormalized_states():
    x = np.linspace(-5, 5, 100)
    states = [np.exp(-(x**2)) for _ in range(3)]
    assert not check_probability_conservation(states, dx=0.1)


def test_probability_conservation_respects_tolerance():
    states = _normalized_states()
    states[2] = states[2] * 1.01
    assert not check_probability_conservation(states, dx=0.1, tolerance=1e-6)
    assert check_probability_conservation(states, dx=0.1, tolerance=0.1)


def test_energy_conservation_true_for_constant_energies():
    energies = np.array([1.5, 1.5, 1.5000000001, 1.4999999999])
    assert energy_conservation(energies)


def test_energy_conservation_false_for_drifting_energies():
    energies = np.array([1.0, 1.5, 2.0, 2.5])
    assert not energy_conservation(energies)


def test_energy_conservation_respects_tolerance():
    energies = np.array([1.0, 1.001, 1.002])
    assert not energy_conservation(energies, tolerance=1e-6)
    assert energy_conservation(energies, tolerance=0.01)


def test_max_probability_error_zero_for_normalized_states():
    states = _normalized_states()
    assert max_probability_error(states, dx=0.1) == pytest.approx(0.0, abs=1e-9)


def test_max_probability_error_finds_worst_state():
    states = _normalized_states()
    states[1] = states[1] * 2.0
    error = max_probability_error(states, dx=0.1)
    assert error == pytest.approx(3.0, rel=1e-6)


def test_max_probability_error_is_nonnegative():
    x = np.linspace(-5, 5, 50)
    states = [np.exp(-(x**2) * (i + 1)) for i in range(3)]
    error = max_probability_error(states, dx=0.1)
    assert error >= 0
