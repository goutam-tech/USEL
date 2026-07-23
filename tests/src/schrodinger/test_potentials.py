from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.potentials import (
    barrier,
    coulomb,
    double_well,
    finite_square_well,
    harmonic_oscillator,
    infinite_square_well,
    kronig_penney,
)


def test_infinite_square_well_zero_inside_infinite_outside():
    x = np.linspace(-5, 5, 21)
    v = infinite_square_well(x, x_min=-2.0, x_max=2.0)
    inside = (x >= -2.0) & (x <= 2.0)
    assert np.all(v[inside] == 0.0)
    assert np.all(np.isinf(v[~inside]))


def test_infinite_square_well_shape_matches_input():
    x = np.linspace(-3, 3, 15)
    v = infinite_square_well(x, x_min=-1.0, x_max=1.0)
    assert v.shape == x.shape


def test_finite_square_well_depth_inside_zero_outside():
    x = np.linspace(-5, 5, 21)
    v = finite_square_well(x, x_min=-2.0, x_max=2.0, depth=3.0)
    inside = (x >= -2.0) & (x <= 2.0)
    assert np.allclose(v[inside], -3.0)
    assert np.allclose(v[~inside], 0.0)


def test_finite_square_well_zero_depth_is_flat():
    x = np.linspace(-5, 5, 21)
    v = finite_square_well(x, x_min=-2.0, x_max=2.0, depth=0.0)
    assert np.allclose(v, 0.0)


def test_harmonic_oscillator_matches_formula():
    x = np.linspace(-5, 5, 11)
    v = harmonic_oscillator(x, omega=2.0, mass=1.5)
    expected = 0.5 * 1.5 * 2.0**2 * x**2
    assert np.allclose(v, expected)


def test_harmonic_oscillator_minimum_at_origin():
    x = np.linspace(-5, 5, 101)
    v = harmonic_oscillator(x)
    assert v[np.argmin(np.abs(x))] == pytest.approx(0.0, abs=1e-10)
    assert np.all(v >= -1e-12)


def test_harmonic_oscillator_default_parameters():
    x = np.array([2.0])
    v = harmonic_oscillator(x)
    assert v[0] == pytest.approx(0.5 * 1.0 * 1.0**2 * 4.0)


def test_double_well_is_symmetric():
    x = np.linspace(-5, 5, 201)
    v = double_well(x, barrier_height=2.0, separation=1.5)
    assert np.allclose(v, v[::-1], atol=1e-10)


def test_double_well_has_two_minima_near_plus_minus_separation():
    x = np.linspace(-5, 5, 2001)
    separation = 2.0
    v = double_well(x, barrier_height=2.0, separation=separation)
    left_region = x < 0
    right_region = x > 0
    x_left_min = x[left_region][np.argmin(v[left_region])]
    x_right_min = x[right_region][np.argmin(v[right_region])]
    assert x_left_min == pytest.approx(-separation, abs=0.1)
    assert x_right_min == pytest.approx(separation, abs=0.1)


def test_barrier_height_inside_zero_outside():
    x = np.linspace(-5, 5, 21)
    v = barrier(x, x_min=-1.0, x_max=1.0, height=7.5)
    inside = (x >= -1.0) & (x <= 1.0)
    assert np.allclose(v[inside], 7.5)
    assert np.allclose(v[~inside], 0.0)


def test_coulomb_sign_follows_charge():
    x = np.linspace(-5, 5, 21)
    v_attractive = coulomb(x, charge=-1.0)
    v_repulsive = coulomb(x, charge=1.0)
    nonzero = x != 0
    assert np.all(v_attractive[nonzero] < 0)
    assert np.all(v_repulsive[nonzero] > 0)


def test_coulomb_is_finite_at_origin():
    x = np.array([0.0])
    v = coulomb(x, charge=-1.0)
    assert np.isfinite(v[0])


def test_coulomb_decays_with_distance():
    x = np.array([1.0, 2.0, 4.0])
    v = coulomb(x, charge=-1.0)
    assert abs(v[0]) > abs(v[1]) > abs(v[2])


def test_kronig_penney_shape_matches_grid():
    x = np.linspace(0, 10, 50)
    v = kronig_penney(x, barrier_height=5.0, period=2.0, width=0.5)
    assert v.shape == x.shape


def test_kronig_penney_only_takes_zero_or_barrier_values():
    x = np.linspace(0, 10, 50)
    v = kronig_penney(x, barrier_height=5.0, period=2.0, width=0.5)
    assert set(np.unique(v)).issubset({0.0, 5.0})


def test_kronig_penney_has_some_nonzero_barrier_regions():
    x = np.linspace(0, 10, 50)
    v = kronig_penney(x, barrier_height=5.0, period=2.0, width=0.5)
    assert np.any(v > 0)
