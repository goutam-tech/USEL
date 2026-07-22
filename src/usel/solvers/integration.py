"""Numerical integration: Trapezoidal, Simpson, and Gaussian Quadrature."""

from __future__ import annotations

from typing import Callable

import numpy as np

from usel.exceptions import ValidationError

Func = Callable[[float], float]


def trapezoidal(f: Func, a: float, b: float, n: int = 1000) -> float:
    """Approximate the integral of ``f`` over [a, b] using the trapezoidal rule.

    Parameters
    ----------
    f: The integrand.
    a, b: Integration bounds.
    n: Number of subintervals (must be >= 1).
    """
    if n < 1:
        raise ValidationError("n must be at least 1")
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    h = (b - a) / n
    return float(h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1])))


def simpson(f: Func, a: float, b: float, n: int = 1000) -> float:
    """Approximate the integral of ``f`` over [a, b] using Simpson's rule.

    ``n`` (number of subintervals) must be even.
    """
    if n < 2:
        raise ValidationError("n must be at least 2")
    if n % 2 != 0:
        n += 1  # silently promote to an even number of subintervals
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    h = (b - a) / n
    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-1:2])
    return float((h / 3.0) * (y[0] + y[-1] + 4 * odd_sum + 2 * even_sum))


def gaussian_quadrature(f: Func, a: float, b: float, n: int = 5) -> float:
    """Approximate the integral of ``f`` over [a, b] using Gauss-Legendre quadrature.

    Parameters
    ----------
    f: The integrand.
    a, b: Integration bounds.
    n: Number of quadrature nodes (degree of precision 2n - 1).
    """
    if n < 1:
        raise ValidationError("n must be at least 1")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # Map nodes/weights from [-1, 1] to [a, b]
    mapped_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    mapped_weights = 0.5 * (b - a) * weights
    total = sum(w * f(x) for w, x in zip(mapped_weights, mapped_nodes))
    return float(total)
