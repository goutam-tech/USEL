from __future__ import annotations

import numpy as np
import pytest

from usel.schrodinger.grids import SpatialGrid, TimeGrid


def test_spatial_grid_builds_correct_linspace():
    grid = SpatialGrid(xmin=-5.0, xmax=5.0, points=101)
    assert len(grid.x) == 101
    assert grid.x[0] == pytest.approx(-5.0)
    assert grid.x[-1] == pytest.approx(5.0)


def test_spatial_grid_spacing_matches_linspace_step():
    grid = SpatialGrid(xmin=0.0, xmax=10.0, points=11)
    assert grid.dx == pytest.approx(1.0)
    assert np.allclose(np.diff(grid.x), grid.dx)


def test_spatial_grid_requires_at_least_two_points():
    with pytest.raises(ValueError):
        SpatialGrid(xmin=0.0, xmax=1.0, points=1)


def test_spatial_grid_rejects_zero_points():
    with pytest.raises(ValueError):
        SpatialGrid(xmin=0.0, xmax=1.0, points=0)


def test_spatial_grid_accessor_methods():
    grid = SpatialGrid(xmin=-2.0, xmax=3.0, points=51)
    assert grid.size() == 51
    assert grid.domain() == (-2.0, 3.0)
    assert grid.spacing() == pytest.approx(grid.dx)


def test_time_grid_builds_correct_linspace():
    tg = TimeGrid(t0=0.0, t_end=2.0, steps=21)
    assert len(tg.t) == 21
    assert tg.t[0] == pytest.approx(0.0)
    assert tg.t[-1] == pytest.approx(2.0)


def test_time_grid_dt_matches_spacing():
    tg = TimeGrid(t0=0.0, t_end=1.0, steps=11)
    assert tg.dt == pytest.approx(0.1)


def test_time_grid_single_step_has_zero_dt():
    tg = TimeGrid(t0=0.0, t_end=5.0, steps=1)
    assert tg.dt == 0
    assert len(tg.t) == 1


def test_time_grid_rejects_nonpositive_steps():
    with pytest.raises(ValueError):
        TimeGrid(t0=0.0, t_end=1.0, steps=0)
    with pytest.raises(ValueError):
        TimeGrid(t0=0.0, t_end=1.0, steps=-3)


def test_time_grid_accessor_methods():
    tg = TimeGrid(t0=0.0, t_end=4.0, steps=41)
    assert tg.size() == 41
    assert tg.timestep() == pytest.approx(tg.dt)
