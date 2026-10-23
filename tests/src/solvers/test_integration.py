import math

import pytest

from usel.exceptions import ValidationError
from usel.solvers.integration import gaussian_quadrature, simpson, trapezoidal


def test_trapezoidal():
    result = trapezoidal(lambda x: x * x, 0, 1)

    assert result == pytest.approx(1 / 3, rel=1e-3)


def test_simpson():
    result = simpson(lambda x: x * x, 0, 1)

    assert result == pytest.approx(1 / 3, rel=1e-6)


def test_gaussian_quadrature():
    result = gaussian_quadrature(lambda x: x * x, 0, 1)

    assert result == pytest.approx(1 / 3, rel=1e-10)


def test_trapezoidal_invalid_n():
    with pytest.raises(ValidationError):
        trapezoidal(lambda x: x, 0, 1, n=0)


def test_simpson_invalid_n():
    with pytest.raises(ValidationError):
        simpson(lambda x: x, 0, 1, n=1)


def test_gaussian_invalid_n():
    with pytest.raises(ValidationError):
        gaussian_quadrature(lambda x: x, 0, 1, n=0)


def test_trapezoidal_sin_over_half_period():
    # ∫ sin(x) dx from 0 to pi = 2
    result = trapezoidal(math.sin, 0, math.pi, n=10_000)
    assert result == pytest.approx(2.0, abs=1e-4)


def test_trapezoidal_exp_over_unit_interval():
    # ∫ exp(x) dx from 0 to 1 = e - 1
    result = trapezoidal(math.exp, 0, 1, n=10_000)
    assert result == pytest.approx(math.e - 1.0, abs=1e-4)


def test_simpson_sin_over_half_period():
    result = simpson(math.sin, 0, math.pi, n=1000)
    assert result == pytest.approx(2.0, abs=1e-6)


def test_simpson_cubic_is_exact():
    # Simpson's rule is exact for cubics: ∫ x^3 dx from 0 to 1 = 1/4
    result = simpson(lambda x: x**3, 0, 1, n=100)
    assert result == pytest.approx(0.25, abs=1e-12)


def test_gaussian_quadrature_sin_over_half_period():
    result = gaussian_quadrature(math.sin, 0, math.pi, n=10)
    assert result == pytest.approx(2.0, abs=1e-10)


def test_trapezoidal_error_decreases_with_n():
    # Trapezoidal error for sin over [0, pi] should shrink as n grows.
    err_coarse = abs(trapezoidal(math.sin, 0, math.pi, n=100) - 2.0)
    err_fine = abs(trapezoidal(math.sin, 0, math.pi, n=1000) - 2.0)
    assert err_fine < err_coarse
