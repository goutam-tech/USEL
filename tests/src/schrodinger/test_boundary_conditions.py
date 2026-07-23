from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.boundary_conditions import (
    BoundaryCondition,
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
)


def test_base_boundary_condition_apply_is_not_implemented():
    bc = BoundaryCondition()
    with pytest.raises(NotImplementedError):
        bc.apply(np.array([1.0, 2.0, 3.0]))


def test_dirichlet_zeroes_the_endpoints():
    psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    result = DirichletBoundary().apply(psi)
    assert result[0] == 0
    assert result[-1] == 0


def test_dirichlet_leaves_interior_untouched():
    psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    result = DirichletBoundary().apply(psi)
    assert np.array_equal(result[1:-1], psi[1:-1])


def test_dirichlet_does_not_mutate_input():
    psi = np.array([1.0, 2.0, 3.0], dtype=complex)
    original = psi.copy()
    DirichletBoundary().apply(psi)
    assert np.array_equal(psi, original)


def test_neumann_copies_neighbouring_values_to_endpoints():
    psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    result = NeumannBoundary().apply(psi)
    assert result[0] == psi[1]
    assert result[-1] == psi[-2]


def test_neumann_leaves_interior_untouched():
    psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    result = NeumannBoundary().apply(psi)
    assert np.array_equal(result[1:-1], psi[1:-1])


def test_neumann_does_not_mutate_input():
    psi = np.array([1.0, 2.0, 3.0], dtype=complex)
    original = psi.copy()
    NeumannBoundary().apply(psi)
    assert np.array_equal(psi, original)


def test_periodic_wraps_last_point_to_first():
    psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    result = PeriodicBoundary().apply(psi)
    assert result[-1] == psi[0]


def test_periodic_leaves_everything_else_untouched():
    psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    result = PeriodicBoundary().apply(psi)
    assert np.array_equal(result[:-1], psi[:-1])


def test_periodic_does_not_mutate_input():
    psi = np.array([1.0, 2.0, 3.0], dtype=complex)
    original = psi.copy()
    PeriodicBoundary().apply(psi)
    assert np.array_equal(psi, original)


def test_all_boundary_subclasses_are_boundary_condition_instances():
    for cls in (DirichletBoundary, NeumannBoundary, PeriodicBoundary):
        assert isinstance(cls(), BoundaryCondition)
