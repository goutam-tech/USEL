from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from usel.exceptions import SolverError
from usel.solvers.root_finding import (
    SolverResult,
    bisection,
    fixed_point,
    gradient_descent,
    newton,
    secant,
)

SQRT2 = math.sqrt(2.0)


def test_solver_result_is_frozen_dataclass():
    result = SolverResult(root=1.0, iterations=1, converged=True, history=[1.0])
    assert result.root == 1.0
    with pytest.raises(FrozenInstanceError):
        result.root = 2.0


def test_newton_finds_sqrt2():
    result = newton(f=lambda x: x**2 - 2, df=lambda x: 2 * x, x0=1.0)
    assert result.converged
    assert result.root == pytest.approx(SQRT2, abs=1e-9)
    assert result.history[0] == 1.0
    assert result.history[-1] == result.root


def test_newton_zero_derivative_raises_solver_error():
    with pytest.raises(SolverError):
        newton(f=lambda x: x**2 - 2, df=lambda x: 2 * x, x0=0.0)


def test_newton_reports_non_convergence_when_capped():
    result = newton(f=lambda x: x**2 - 2, df=lambda x: 2 * x, x0=1000.0, max_iter=1)
    assert not result.converged
    assert result.iterations == 1


def test_newton_history_grows_by_one_each_iteration():
    result = newton(f=lambda x: x**2 - 2, df=lambda x: 2 * x, x0=1.0)
    assert len(result.history) == result.iterations + 1


def test_bisection_finds_sqrt2():
    result = bisection(f=lambda x: x**2 - 2, a=0.0, b=2.0)
    assert result.converged
    assert result.root == pytest.approx(SQRT2, abs=1e-9)


def test_bisection_same_sign_endpoints_raise_solver_error():
    with pytest.raises(SolverError):
        bisection(f=lambda x: x**2 + 1, a=-1.0, b=1.0)


def test_bisection_root_stays_within_bracket():
    result = bisection(f=lambda x: x**3 - x - 2, a=1.0, b=2.0)
    assert 1.0 <= result.root <= 2.0
    assert result.converged


def test_secant_finds_sqrt2():
    result = secant(f=lambda x: x**2 - 2, x0=1.0, x1=2.0)
    assert result.converged
    assert result.root == pytest.approx(SQRT2, abs=1e-9)


def test_secant_zero_denominator_raises_solver_error():
    with pytest.raises(SolverError):
        secant(f=lambda x: 5.0, x0=0.0, x1=1.0)


def test_secant_initial_history_contains_seed_points():
    result = secant(f=lambda x: x**2 - 2, x0=1.0, x1=2.0)
    assert result.history[0] == 1.0
    assert result.history[1] == 2.0


def test_gradient_descent_minimizes_simple_quadratic():
    result = gradient_descent(df=lambda x: 2 * (x - 3), x0=0.0, learning_rate=0.1)
    assert result.converged
    assert result.root == pytest.approx(3.0, abs=1e-6)


def test_gradient_descent_respects_max_iter_when_slow():
    result = gradient_descent(df=lambda x: 2 * (x - 3), x0=0.0, learning_rate=1e-6, max_iter=5)
    assert not result.converged
    assert result.iterations == 5


def test_fixed_point_cosine_converges_to_dottie_number():
    result = fixed_point(g=math.cos, x0=0.5)
    assert result.converged
    assert result.root == pytest.approx(0.7390851332, abs=1e-8)


def test_fixed_point_identity_converges_immediately():
    result = fixed_point(g=lambda x: x, x0=3.14)
    assert result.converged
    assert result.iterations == 1
    assert result.root == pytest.approx(3.14)


def test_fixed_point_non_convergent_map_reports_failure():
    result = fixed_point(g=lambda x: 2 * x, x0=1.0, max_iter=10)
    assert not result.converged
    assert result.iterations == 10
