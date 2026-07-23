from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.wavefunctions import (
    double_gaussian,
    gaussian_wavepacket,
    harmonic_oscillator_state,
    particle_in_box_state,
)


def test_particle_in_box_ground_state_is_normalized():
    x = np.linspace(0, 10, 2000)
    dx = x[1] - x[0]
    psi = particle_in_box_state(x, n=1, x_min=2.0, x_max=8.0)
    norm = np.sum(np.abs(psi) ** 2) * dx
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_particle_in_box_state_is_zero_outside_well():
    x = np.linspace(0, 10, 500)
    psi = particle_in_box_state(x, n=1, x_min=2.0, x_max=8.0)
    outside = (x < 2.0) | (x > 8.0)
    assert np.all(psi[outside] == 0.0)


def test_particle_in_box_state_matches_analytic_formula_inside():
    x = np.linspace(2.0, 8.0, 300)
    n = 2
    x_min, x_max = 2.0, 8.0
    L = x_max - x_min
    psi = particle_in_box_state(x, n=n, x_min=x_min, x_max=x_max)
    expected = np.sqrt(2.0 / L) * np.sin(n * np.pi * (x - x_min) / L)
    assert np.allclose(psi, expected)


def test_particle_in_box_higher_states_have_more_nodes():
    x = np.linspace(0, 10, 4000)
    psi2 = particle_in_box_state(x, n=2, x_min=2.0, x_max=8.0)
    psi3 = particle_in_box_state(x, n=3, x_min=2.0, x_max=8.0)
    sign_changes_2 = np.sum(np.diff(np.sign(psi2[psi2 != 0])) != 0)
    sign_changes_3 = np.sum(np.diff(np.sign(psi3[psi3 != 0])) != 0)
    assert sign_changes_3 > sign_changes_2


def test_harmonic_oscillator_ground_state_is_normalized():
    x = np.linspace(-8, 8, 2000)
    dx = x[1] - x[0]
    psi = harmonic_oscillator_state(x, n=0)
    norm = np.sum(np.abs(psi) ** 2) * dx
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_harmonic_oscillator_ground_state_matches_gaussian_formula():
    x = np.linspace(-8, 8, 500)
    mass, omega = 1.0, 1.0
    psi = harmonic_oscillator_state(x, n=0, omega=omega, mass=mass)
    expected = (mass * omega / np.pi) ** 0.25 * np.exp(-0.5 * mass * omega * x**2)
    assert np.allclose(psi, expected, atol=1e-10)


def test_harmonic_oscillator_states_are_orthogonal():
    x = np.linspace(-8, 8, 4000)
    dx = x[1] - x[0]
    psi0 = harmonic_oscillator_state(x, n=0)
    psi1 = harmonic_oscillator_state(x, n=1)
    overlap = np.sum(psi0 * psi1) * dx
    assert overlap == pytest.approx(0.0, abs=1e-6)


def test_harmonic_oscillator_first_excited_state_is_odd():
    x = np.linspace(-8, 8, 500)
    psi1 = harmonic_oscillator_state(x, n=1)
    assert np.allclose(psi1, -psi1[::-1], atol=1e-10)


def test_harmonic_oscillator_ground_state_is_even():
    x = np.linspace(-8, 8, 500)
    psi0 = harmonic_oscillator_state(x, n=0)
    assert np.allclose(psi0, psi0[::-1], atol=1e-10)


def test_gaussian_wavepacket_is_normalized():
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    psi = gaussian_wavepacket(x, x0=1.0, sigma=0.5, k0=3.0)
    norm = np.sum(np.abs(psi) ** 2) * dx
    assert norm == pytest.approx(1.0, abs=1e-8)


def test_gaussian_wavepacket_centered_correctly():
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    psi = gaussian_wavepacket(x, x0=2.0, sigma=0.5, k0=0.0)
    mean_x = np.sum(x * np.abs(psi) ** 2) * dx
    assert mean_x == pytest.approx(2.0, abs=1e-2)


def test_gaussian_wavepacket_zero_momentum_is_real():
    x = np.linspace(-10, 10, 500)
    psi = gaussian_wavepacket(x, x0=0.0, sigma=1.0, k0=0.0)
    assert np.allclose(psi.imag, 0.0, atol=1e-10)


def test_double_gaussian_is_normalized():
    x = np.linspace(-10, 10, 4000)
    dx = x[1] - x[0]
    psi = double_gaussian(x)
    norm = np.sum(np.abs(psi) ** 2) * dx
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_double_gaussian_has_density_peaks_near_both_centers():
    x = np.linspace(-10, 10, 4000)
    x1, x2, sigma = -2.0, 2.0, 0.3
    psi = double_gaussian(x, x1=x1, x2=x2, sigma=sigma, k1=0.0, k2=0.0)
    density = np.abs(psi) ** 2
    near_x1 = density[np.argmin(np.abs(x - x1))]
    near_x2 = density[np.argmin(np.abs(x - x2))]
    far_from_both = density[np.argmin(np.abs(x - 0.0))]
    assert near_x1 > far_from_both
    assert near_x2 > far_from_both
