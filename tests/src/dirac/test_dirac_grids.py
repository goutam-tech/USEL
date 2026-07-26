import numpy as np
import pytest

from usel.dirac.grids import DiracGrid


def test_grid_creation():
    grid = DiracGrid(-5, 5, 100)

    assert grid.size == 100
    assert grid.x[0] == -5
    assert grid.x[-1] == 5


def test_grid_spacing():
    grid = DiracGrid(0, 10, 11)

    assert grid.dx == 1


def test_grid_requires_minimum_points():
    with pytest.raises(ValueError):
        DiracGrid(0, 1, 2)


def test_index_returns_nearest_point():
    grid = DiracGrid(0, 10, 11)

    index = grid.index(4.2)

    assert index == 4


def test_zero_field_shape():
    grid = DiracGrid(0, 1, 20)

    field = grid.zeros()

    assert field.shape == (20,)
    assert np.all(field == 0)


def test_spinor_zero_shape():
    grid = DiracGrid(0, 1, 20)

    psi = grid.spinor_zeros()

    assert psi.shape == (4, 20)

    assert psi.dtype == np.complex128
