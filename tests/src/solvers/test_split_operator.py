from __future__ import annotations

import numpy as np
import pytest

from usel.solvers.split_operator import evolve


class Grid:
    def __init__(self, x: np.ndarray):
        self.x = x
        self.dx = x[1] - x[0]


def make_grid(n=512, extent=20.0):
    x = np.linspace(-extent, extent, n, endpoint=False)
    return Grid(x)


def gaussian_wavepacket(x, sigma0=1.0, k0=2.0, x0=0.0):
    envelope = (1.0 / (2.0 * np.pi * sigma0**2)) ** 0.25 * np.exp(
        -((x - x0) ** 2) / (4.0 * sigma0**2)
    )
    return (envelope * np.exp(1j * k0 * x)).astype(complex)


def norm(psi, dx):
    return float(np.sum(np.abs(psi) ** 2) * dx)


def variance(psi, x, dx):
    prob = np.abs(psi) ** 2 * dx
    mean = np.sum(x * prob)
    return float(np.sum((x - mean) ** 2 * prob))


def test_output_shape_matches_steps_and_grid_size():
    grid = make_grid(n=256)
    psi0 = gaussian_wavepacket(grid.x)
    V = np.zeros_like(grid.x)
    states = evolve(psi0, V, grid, dt=0.01, steps=40)
    assert states.shape == (40, 256)


def test_does_not_mutate_input_state():
    grid = make_grid(n=128)
    psi0 = gaussian_wavepacket(grid.x)
    psi0_copy = psi0.copy()
    V = np.zeros_like(grid.x)
    evolve(psi0, V, grid, dt=0.01, steps=10)
    assert np.array_equal(psi0, psi0_copy)


def test_zero_steps_returns_empty_array():
    grid = make_grid(n=64)
    psi0 = gaussian_wavepacket(grid.x)
    V = np.zeros_like(grid.x)
    states = evolve(psi0, V, grid, dt=0.01, steps=0)
    assert states.shape == (0,)


def test_norm_is_conserved_free_particle():
    grid = make_grid(n=512)
    psi0 = gaussian_wavepacket(grid.x)
    V = np.zeros_like(grid.x)

    states = evolve(psi0, V, grid, dt=0.01, steps=200)
    norms = np.array([norm(s, grid.dx) for s in states])

    assert norms[0] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_norm_is_conserved_with_nonzero_potential():
    grid = make_grid(n=512)
    psi0 = gaussian_wavepacket(grid.x)
    m, omega = 1.0, 1.0
    V = 0.5 * m * omega**2 * grid.x**2

    states = evolve(psi0, V, grid, dt=0.01, steps=300, mass=m)
    norms = np.array([norm(s, grid.dx) for s in states])

    assert np.allclose(norms, 1.0, atol=1e-6)


def test_free_gaussian_wavepacket_spreads_according_to_analytic_formula():
    grid = make_grid(n=1024, extent=40.0)
    sigma0 = 1.0
    mass, hbar = 1.0, 1.0
    psi0 = gaussian_wavepacket(grid.x, sigma0=sigma0, k0=1.5)
    V = np.zeros_like(grid.x)

    dt, steps = 0.01, 200
    states = evolve(psi0, V, grid, dt=dt, steps=steps, mass=mass, hbar=hbar)

    t_final = dt * steps
    var_numeric = variance(states[-1], grid.x, grid.dx)
    var_analytic = sigma0**2 + (hbar * t_final / (2.0 * mass * sigma0)) ** 2

    assert var_numeric == pytest.approx(var_analytic, rel=1e-3)


def test_harmonic_ground_state_is_stationary_up_to_global_phase():
    grid = make_grid(n=1024, extent=15.0)
    mass, omega, hbar = 1.0, 1.0, 1.0
    psi0 = (mass * omega / (np.pi * hbar)) ** 0.25 * np.exp(
        -mass * omega * grid.x**2 / (2.0 * hbar)
    )
    psi0 = psi0.astype(complex)
    V = 0.5 * mass * omega**2 * grid.x**2

    dt, steps = 0.01, 300
    states = evolve(psi0, V, grid, dt=dt, steps=steps, mass=mass, hbar=hbar)

    assert np.allclose(np.abs(states[-1]), np.abs(psi0), atol=1e-5)

    e0 = 0.5 * hbar * omega
    t_final = dt * steps
    expected_phase = np.exp(-1j * e0 * t_final / hbar)
    overlap = np.vdot(psi0, states[-1]) * grid.dx
    assert overlap == pytest.approx(expected_phase, abs=1e-5)


def test_finer_dt_more_closely_matches_analytic_spreading():
    grid = make_grid(n=1024, extent=40.0)
    sigma0 = 1.0
    psi0 = gaussian_wavepacket(grid.x, sigma0=sigma0, k0=1.5)
    V = np.zeros_like(grid.x)
    t_final = 2.0

    def spreading_error(dt):
        steps = int(round(t_final / dt))
        states = evolve(psi0, V, grid, dt=dt, steps=steps)
        var_numeric = variance(states[-1], grid.x, grid.dx)
        var_analytic = sigma0**2 + (t_final / (2.0 * sigma0)) ** 2
        return abs(var_numeric - var_analytic)

    err_coarse = spreading_error(0.05)
    err_fine = spreading_error(0.01)

    assert err_fine <= err_coarse + 1e-9
