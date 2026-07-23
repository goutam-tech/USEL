from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.hamiltonian import Hamiltonian, build_hamiltonian, solve_energy_levels
from usel.schrodinger.operators import (
    QuantumOperator,
    kinetic_energy_operator,
    potential_operator,
)
from usel.schrodinger.potentials import harmonic_oscillator


def test_hamiltonian_matrix_is_sum_of_kinetic_and_potential():
    grid = SpatialGrid(-5.0, 5.0, 50)
    kinetic = kinetic_energy_operator(grid)
    potential = potential_operator(np.ones(50))
    ham = Hamiltonian(kinetic, potential)
    expected = kinetic.matrix + potential.matrix
    assert np.allclose(ham.matrix.toarray(), expected.toarray())


def test_hamiltonian_is_a_quantum_operator():
    grid = SpatialGrid(-5.0, 5.0, 20)
    kinetic = kinetic_energy_operator(grid)
    potential = potential_operator(np.zeros(20))
    ham = Hamiltonian(kinetic, potential)
    assert isinstance(ham, QuantumOperator)
    assert ham.shape == (20, 20)


def test_hamiltonian_is_hermitian_for_real_potential():
    grid = SpatialGrid(-5.0, 5.0, 60)
    v = harmonic_oscillator(grid.x)
    kinetic = kinetic_energy_operator(grid)
    potential = potential_operator(v)
    ham = Hamiltonian(kinetic, potential)
    dense = ham.matrix.toarray()
    assert np.allclose(dense, dense.conj().T)


def test_build_hamiltonian_matches_manual_construction():
    grid = SpatialGrid(-5.0, 5.0, 50)
    v = harmonic_oscillator(grid.x)
    ham = build_hamiltonian(grid, v)

    kinetic = kinetic_energy_operator(grid)
    potential = potential_operator(v)
    expected = kinetic.matrix + potential.matrix

    assert np.allclose(ham.matrix.toarray(), expected.toarray())


def test_build_hamiltonian_respects_mass_and_hbar():
    grid = SpatialGrid(-5.0, 5.0, 30)
    v = np.zeros(30)
    ham_light = build_hamiltonian(grid, v, mass=1.0, hbar=1.0)
    ham_heavy = build_hamiltonian(grid, v, mass=4.0, hbar=1.0)
    # heavier mass -> smaller kinetic energy matrix elements
    assert np.max(np.abs(ham_heavy.matrix.toarray())) < np.max(np.abs(ham_light.matrix.toarray()))


def test_harmonic_oscillator_energy_levels_match_theory():
    """E_n = hbar * omega * (n + 1/2) for the QHO in natural units."""
    grid = SpatialGrid(-8.0, 8.0, 400)
    v = harmonic_oscillator(grid.x, omega=1.0, mass=1.0)
    ham = build_hamiltonian(grid, v, mass=1.0, hbar=1.0)
    energies, states = solve_energy_levels(ham, levels=5)

    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(energies, expected, atol=0.01)


def test_solve_energy_levels_returns_sorted_ascending_energies():
    grid = SpatialGrid(-8.0, 8.0, 300)
    v = harmonic_oscillator(grid.x)
    ham = build_hamiltonian(grid, v)
    energies, _ = solve_energy_levels(ham, levels=4)
    assert np.all(np.diff(energies) > 0)


def test_solve_energy_levels_eigenstate_shape():
    grid = SpatialGrid(-8.0, 8.0, 200)
    v = harmonic_oscillator(grid.x)
    ham = build_hamiltonian(grid, v)
    energies, states = solve_energy_levels(ham, levels=3)
    assert energies.shape == (3,)
    assert states.shape == (200, 3)


def test_infinite_square_well_ground_state_energy_matches_theory():
    """E_1 = pi^2 hbar^2 / (2 m L^2) for the infinite square well, using a
    large but finite potential to keep the finite-difference matrix
    well-conditioned (true np.inf would break the linear algebra)."""
    x_min, x_max = 0.0, np.pi
    grid = SpatialGrid(x_min, x_max, 300)
    large_value = 1e6
    v = np.where((grid.x > x_min) & (grid.x < x_max), 0.0, large_value)
    ham = build_hamiltonian(grid, v)
    energies, _ = solve_energy_levels(ham, levels=1)

    L = x_max - x_min
    expected_ground = np.pi**2 / (2.0 * L**2)
    assert energies[0] == pytest.approx(expected_ground, rel=0.05)
