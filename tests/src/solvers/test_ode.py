from __future__ import annotations

import math

import numpy as np
import pytest

from usel.exceptions import ValidationError
from usel.solvers.ode import euler, improved_euler, rk2, rk4

ALL_METHODS = pytest.mark.parametrize(
    "method",
    [euler, improved_euler, rk2, rk4],
    ids=["euler", "improved_euler", "rk2", "rk4"],
)


@ALL_METHODS
def test_output_shapes_and_endpoints(method):
    t0, y0, t_end, h = 0.0, 1.0, 2.0, 0.1
    t, y = method(lambda t, y: y, t0, y0, t_end, h)

    n_steps = round((t_end - t0) / h)
    assert len(t) == n_steps + 1
    assert len(y) == n_steps + 1
    assert t[0] == pytest.approx(t0)
    assert t[-1] == pytest.approx(t_end)
    assert y[0] == y0


@ALL_METHODS
def test_time_grid_is_uniform(method):
    t, _ = method(lambda t, y: 0.0, 0.0, 0.0, 1.0, 0.25)
    diffs = np.diff(t)
    assert np.allclose(diffs, 0.25)


@ALL_METHODS
def test_constant_derivative_is_exact(method):
    """dy/dt = c has an exact linear solution every method should reproduce."""
    c = 3.5
    t, y = method(lambda t, y: c, 0.0, 1.0, 1.0, 0.05)
    expected = 1.0 + c * t
    assert np.allclose(y, expected, atol=1e-10)


@ALL_METHODS
@pytest.mark.parametrize("h", [0.0, -0.1])
def test_nonpositive_step_raises(method, h):
    with pytest.raises(ValidationError):
        method(lambda t, y: y, 0.0, 1.0, 1.0, h)


@ALL_METHODS
@pytest.mark.parametrize("t0,t_end", [(1.0, 1.0), (1.0, 0.5)])
def test_t_end_not_after_t0_raises(method, t0, t_end):
    with pytest.raises(ValidationError):
        method(lambda t, y: y, t0, 1.0, t_end, 0.1)


@ALL_METHODS
def test_exponential_growth_matches_analytic(method):
    """dy/dt = y, y(0) = 1  =>  y(t) = e^t."""
    t, y = method(lambda t, y: y, 0.0, 1.0, 1.0, 1e-3)
    analytic = np.exp(t)
    # h = 1e-3 is small enough that even first-order Euler is close.
    assert np.allclose(y, analytic, atol=2e-3)


@ALL_METHODS
def test_exponential_decay_matches_analytic(method):
    """dy/dt = -2y, y(0) = 5  =>  y(t) = 5 e^{-2t}."""
    t, y = method(lambda t, y: -2.0 * y, 0.0, 5.0, 3.0, 1e-3)
    analytic = 5.0 * np.exp(-2.0 * t)
    assert np.allclose(y, analytic, atol=5e-3)


@ALL_METHODS
def test_nonhomogeneous_linear_ode(method):
    """dy/dt = -y + t, y(0) = 1  =>  y(t) = 2e^{-t} + t - 1."""
    t, y = method(lambda t, y: -y + t, 0.0, 1.0, 2.0, 1e-3)
    analytic = 2.0 * np.exp(-t) + t - 1.0
    assert np.allclose(y, analytic, atol=5e-3)


def test_rk4_is_dramatically_more_accurate_than_euler():
    """At a moderate step size RK4's global error should be orders of
    magnitude smaller than forward Euler's, for the same h."""
    h = 0.05
    t_euler, y_euler = euler(lambda t, y: y, 0.0, 1.0, 1.0, h)
    t_rk4, y_rk4 = rk4(lambda t, y: y, 0.0, 1.0, 1.0, h)

    err_euler = abs(y_euler[-1] - math.e)
    err_rk4 = abs(y_rk4[-1] - math.e)

    assert err_rk4 < err_euler
    assert err_rk4 < 1e-6
    assert err_euler > 1e-2


@pytest.mark.parametrize(
    "method,min_ratio",
    [
        (euler, 1.8),
        (improved_euler, 3.5),
        (rk2, 3.5),
        (rk4, 13.0),
    ],
    ids=["euler-order1", "improved_euler-order2", "rk2-order2", "rk4-order4"],
)
def test_empirical_convergence_order(method, min_ratio):
    def f(t, y):
        return y

    analytic_end = math.e

    def global_error(h):
        _, y = method(f, 0.0, 1.0, 1.0, h)
        return abs(y[-1] - analytic_end)

    h = 0.02
    err_coarse = global_error(h)
    err_fine = global_error(h / 2.0)

    assert err_fine > 0
    ratio = err_coarse / err_fine
    assert ratio >= min_ratio, (
        f"{method.__name__}: expected error to shrink by >= {min_ratio}x "
        f"when halving h, got {ratio:.2f}x"
    )


def test_validate_step_computes_expected_n_steps():
    from usel.solvers.ode import _validate_step

    assert _validate_step(0.0, 1.0, 0.1) == 10
    assert _validate_step(0.0, 1.0, 0.3) == round(1.0 / 0.3)
