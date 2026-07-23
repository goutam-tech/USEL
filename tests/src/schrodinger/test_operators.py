from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.operators import (
    QuantumOperator,
    kinetic_energy_operator,
    momentum_operator,
    position_operator,
    potential_operator,
)


def test_quantum_operator_apply_is_matrix_vector_product():
    matrix = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    op = QuantumOperator(matrix)
    result = op.apply(np.array([1.0, 1.0]))
    assert np.allclose(result, [3.0, 7.0])


def test_quantum_operator_shape_property():
    matrix = csr_matrix(np.eye(5))
    op = QuantumOperator(matrix)
    assert op.shape == (5, 5)


def test_position_operator_matches_grid_values():
    grid = SpatialGrid(-5.0, 5.0, 50)
    op = position_operator(grid)
    psi = np.exp(-(grid.x**2))
    result = op.apply(psi)
    assert np.allclose(result, grid.x * psi)


def test_position_operator_is_diagonal_and_hermitian():
    grid = SpatialGrid(-3.0, 3.0, 20)
    op = position_operator(grid)
    dense = op.matrix.toarray()
    assert np.allclose(dense, np.diag(grid.x))
    assert np.allclose(dense, dense.conj().T)


def test_momentum_operator_is_hermitian():
    grid = SpatialGrid(-5.0, 5.0, 100)
    op = momentum_operator(grid)
    dense = op.matrix.toarray()
    assert np.allclose(dense, dense.conj().T)


def test_momentum_operator_eigenvalue_on_plane_wave_interior():
    """p psi ~ hbar k psi for psi = e^{ikx}, away from the boundary where
    the central-difference stencil has no neighbour to reference."""
    grid = SpatialGrid(-10.0, 10.0, 400)
    k = 1.3
    psi = np.exp(1j * k * grid.x)
    op = momentum_operator(grid, hbar=1.0)
    result = op.apply(psi)
    expected = k * psi
    interior = slice(10, -10)
    assert np.allclose(result[interior], expected[interior], atol=1e-3)


def test_momentum_operator_scales_with_hbar():
    grid = SpatialGrid(-5.0, 5.0, 50)
    op1 = momentum_operator(grid, hbar=1.0)
    op2 = momentum_operator(grid, hbar=2.0)
    assert np.allclose(op2.matrix.toarray(), 2.0 * op1.matrix.toarray())


def test_kinetic_energy_operator_is_hermitian():
    grid = SpatialGrid(-5.0, 5.0, 100)
    op = kinetic_energy_operator(grid)
    dense = op.matrix.toarray()
    assert np.allclose(dense, dense.conj().T)


def test_kinetic_energy_operator_matches_analytic_second_derivative():
    """T psi = -(hbar^2 / 2m) psi'' for a Gaussian, checked away from the
    grid boundary where the 3-point stencil is well defined."""
    grid = SpatialGrid(-10.0, 10.0, 400)
    mass, hbar, sigma = 1.0, 1.0, 1.0
    psi = np.exp(-(grid.x**2) / (2 * sigma**2))
    op = kinetic_energy_operator(grid, mass=mass, hbar=hbar)
    result = op.apply(psi)

    d2psi_analytic = ((grid.x**2 / sigma**4) - 1.0 / sigma**2) * psi
    expected = -(hbar**2 / (2 * mass)) * d2psi_analytic

    interior = slice(10, -10)
    assert np.allclose(result[interior], expected[interior], atol=1e-3)


def test_kinetic_energy_operator_scales_inversely_with_mass():
    grid = SpatialGrid(-5.0, 5.0, 50)
    op1 = kinetic_energy_operator(grid, mass=1.0)
    op2 = kinetic_energy_operator(grid, mass=2.0)
    assert np.allclose(op1.matrix.toarray(), 2.0 * op2.matrix.toarray())


def test_potential_operator_is_diagonal_multiplication():
    v = np.array([0.0, 1.0, 4.0, 9.0])
    op = potential_operator(v)
    psi = np.array([1.0, 1.0, 1.0, 1.0])
    result = op.apply(psi)
    assert np.allclose(result, v)


def test_potential_operator_matrix_is_diagonal():
    v = np.array([1.0, 2.0, 3.0])
    op = potential_operator(v)
    assert np.allclose(op.matrix.toarray(), np.diag(v))
