from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.observables import (
    expectation,
    fidelity,
    overlap,
    probability,
    uncertainty,
    variance,
)
from usel.schrodinger.operators import position_operator


def gaussian(x, x0=0.0, sigma=1.0):
    psi = np.exp(-((x - x0) ** 2) / (4 * sigma**2))
    dx = x[1] - x[0]
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


def test_expectation_of_position_on_centered_gaussian_is_zero():
    grid = SpatialGrid(-10.0, 10.0, 2000)
    psi = gaussian(grid.x, x0=0.0, sigma=1.0)
    x_op = position_operator(grid)
    result = expectation(psi, x_op, grid.dx)
    assert result == pytest.approx(0.0, abs=1e-8)


def test_expectation_of_position_on_shifted_gaussian_matches_center():
    grid = SpatialGrid(-10.0, 10.0, 2000)
    psi = gaussian(grid.x, x0=2.5, sigma=0.7)
    x_op = position_operator(grid)
    result = expectation(psi, x_op, grid.dx)
    assert result.real == pytest.approx(2.5, abs=1e-3)


def test_variance_of_position_on_gaussian_matches_sigma_squared():
    grid = SpatialGrid(-15.0, 15.0, 4000)
    sigma = 1.2
    psi = gaussian(grid.x, x0=0.0, sigma=sigma)
    x_op = position_operator(grid)
    result = variance(psi, x_op, grid.dx)
    assert result == pytest.approx(sigma**2, rel=1e-2)


def test_uncertainty_is_sqrt_of_variance():
    grid = SpatialGrid(-15.0, 15.0, 4000)
    psi = gaussian(grid.x, x0=1.0, sigma=0.9)
    x_op = position_operator(grid)
    var = variance(psi, x_op, grid.dx)
    unc = uncertainty(psi, x_op, grid.dx)
    assert unc == pytest.approx(np.sqrt(var))


def test_variance_is_nonnegative_for_hermitian_operator():
    grid = SpatialGrid(-10.0, 10.0, 500)
    psi = gaussian(grid.x, x0=-1.0, sigma=2.0)
    x_op = position_operator(grid)
    assert variance(psi, x_op, grid.dx) >= 0


def test_overlap_of_identical_normalized_state_is_one():
    grid = SpatialGrid(-10.0, 10.0, 1000)
    psi = gaussian(grid.x)
    result = overlap(psi, psi, grid.dx)
    assert result.real == pytest.approx(1.0, abs=1e-6)


def test_overlap_of_well_separated_gaussians_is_near_zero():
    grid = SpatialGrid(-30.0, 30.0, 4000)
    psi1 = gaussian(grid.x, x0=-10.0, sigma=0.5)
    psi2 = gaussian(grid.x, x0=10.0, sigma=0.5)
    result = overlap(psi1, psi2, grid.dx)
    assert abs(result) < 1e-6


def test_fidelity_of_identical_state_is_one():
    grid = SpatialGrid(-10.0, 10.0, 1000)
    psi = gaussian(grid.x)
    assert fidelity(psi, psi, grid.dx) == pytest.approx(1.0, abs=1e-6)


def test_fidelity_of_orthogonal_states_is_zero():
    grid = SpatialGrid(-30.0, 30.0, 4000)
    psi1 = gaussian(grid.x, x0=-10.0, sigma=0.5)
    psi2 = gaussian(grid.x, x0=10.0, sigma=0.5)
    assert fidelity(psi1, psi2, grid.dx) < 1e-10


def test_fidelity_is_squared_modulus_of_overlap():
    grid = SpatialGrid(-10.0, 10.0, 1000)
    psi1 = gaussian(grid.x, x0=0.0)
    psi2 = gaussian(grid.x, x0=0.3)
    f = fidelity(psi1, psi2, grid.dx)
    ov = overlap(psi1, psi2, grid.dx)
    assert f == pytest.approx(abs(ov) ** 2)


def test_probability_with_no_region_returns_full_density():
    psi = np.array([1.0 + 1.0j, 2.0, 0.0])
    result = probability(psi)
    assert np.allclose(result, np.abs(psi) ** 2)


def test_probability_with_region_sums_selected_density():
    psi = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
    density = np.abs(psi) ** 2
    region = slice(1, 3)
    result = probability(psi, region=region)
    assert result == pytest.approx(np.sum(density[1:3]))


def test_probability_with_boolean_mask_region():
    psi = np.array([1.0, 2.0, 3.0], dtype=complex)
    mask = np.array([True, False, True])
    result = probability(psi, region=mask)
    assert result == pytest.approx(1.0 + 9.0)
