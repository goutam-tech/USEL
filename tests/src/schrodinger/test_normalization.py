from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.normalization import (
    is_normalized,
    norm,
    normalize,
    probability_density,
)


def test_norm_of_already_normalized_gaussian():
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    sigma = 1.0
    psi = (1 / (2 * np.pi * sigma**2)) ** 0.25 * np.exp(-(x**2) / (4 * sigma**2))
    assert norm(psi, dx) == pytest.approx(1.0, abs=1e-4)


def test_norm_scales_linearly_with_amplitude():
    x = np.linspace(-5, 5, 500)
    dx = x[1] - x[0]
    psi = np.exp(-(x**2))
    base_norm = norm(psi, dx)
    scaled_norm = norm(3.0 * psi, dx)
    assert scaled_norm == pytest.approx(3.0 * base_norm)


def test_norm_of_zero_wavefunction_is_zero():
    psi = np.zeros(10, dtype=complex)
    assert norm(psi, dx=0.1) == 0.0


def test_normalize_produces_unit_norm():
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    psi = np.exp(-((x - 1.0) ** 2))
    normalized = normalize(psi, dx)
    assert norm(normalized, dx) == pytest.approx(1.0, abs=1e-8)


def test_normalize_preserves_relative_phase_and_shape():
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    psi = np.exp(-(x**2)) * np.exp(1j * 2.0 * x)
    normalized = normalize(psi, dx)
    # ratio between any two components should be unchanged by a real scalar rescale
    ratio_before = psi[100] / psi[200]
    ratio_after = normalized[100] / normalized[200]
    assert ratio_after == pytest.approx(ratio_before, rel=1e-10)


def test_normalize_zero_wavefunction_raises_value_error():
    psi = np.zeros(10, dtype=complex)
    with pytest.raises(ValueError):
        normalize(psi, dx=0.1)


def test_is_normalized_true_for_normalized_state():
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    psi = normalize(np.exp(-(x**2)), dx)
    assert is_normalized(psi, dx)


def test_is_normalized_false_for_unnormalized_state():
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    psi = np.exp(-(x**2))  # not normalized (peak amplitude 1)
    assert not is_normalized(psi, dx)


def test_is_normalized_respects_tolerance_argument():
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    psi = normalize(np.exp(-(x**2)), dx) * 1.001  # slightly off from unit norm
    assert not is_normalized(psi, dx, tolerance=1e-6)
    assert is_normalized(psi, dx, tolerance=1e-1)


def test_probability_density_is_squared_modulus():
    psi = np.array([1.0 + 1.0j, 2.0, 0.0])
    density = probability_density(psi)
    assert np.allclose(density, [2.0, 4.0, 0.0])


def test_probability_density_is_always_nonnegative():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=50) + 1j * rng.normal(size=50)
    density = probability_density(psi)
    assert np.all(density >= 0)
