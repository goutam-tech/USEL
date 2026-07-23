"""Root-finding and optimization methods: Newton, Bisection, Secant,
Gradient Descent, and Fixed Point Iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from usel.exceptions import SolverError

Scalar = float
ScalarFunc = Callable[[Scalar], Scalar]


@dataclass(frozen=True)
class SolverResult:
    """Result of a root-finding or optimization routine."""

    root: float
    iterations: int
    converged: bool
    history: list[float]


def newton(
    f: ScalarFunc,
    df: ScalarFunc,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> SolverResult:
    """Newton-Raphson root finding for ``f(x) = 0`` given its derivative ``df``."""
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise SolverError(f"Derivative is zero at x={x}; Newton's method cannot continue")
        x_next = x - fx / dfx
        history.append(x_next)
        if abs(x_next - x) < tol:
            return SolverResult(root=x_next, iterations=i, converged=True, history=history)
        x = x_next
    return SolverResult(root=x, iterations=max_iter, converged=False, history=history)


def bisection(
    f: ScalarFunc,
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> SolverResult:
    """Bisection method for root finding on a bracketing interval [a, b]."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise SolverError("f(a) and f(b) must have opposite signs for bisection")

    history = []
    for i in range(1, max_iter + 1):
        midpoint = (a + b) / 2.0
        f_mid = f(midpoint)
        history.append(midpoint)
        if abs(f_mid) < tol or (b - a) / 2.0 < tol:
            return SolverResult(root=midpoint, iterations=i, converged=True, history=history)
        if fa * f_mid < 0:
            b, fb = midpoint, f_mid
        else:
            a, fa = midpoint, f_mid
    return SolverResult(root=(a + b) / 2.0, iterations=max_iter, converged=False, history=history)


def secant(
    f: ScalarFunc,
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> SolverResult:
    """Secant method for root finding using two initial guesses."""
    history = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for i in range(1, max_iter + 1):
        if f1 - f0 == 0:
            raise SolverError("Zero denominator encountered in secant method")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            return SolverResult(root=x2, iterations=i, converged=True, history=history)
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    return SolverResult(root=x1, iterations=max_iter, converged=False, history=history)


def gradient_descent(
    df: ScalarFunc,
    x0: float,
    learning_rate: float = 0.01,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> SolverResult:
    """Gradient descent minimization given the derivative ``df`` of a 1D function."""
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        grad = df(x)
        x_next = x - learning_rate * grad
        history.append(x_next)
        if abs(x_next - x) < tol:
            return SolverResult(root=x_next, iterations=i, converged=True, history=history)
        x = x_next
    return SolverResult(root=x, iterations=max_iter, converged=False, history=history)


def fixed_point(
    g: ScalarFunc,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult:
    """Fixed-point iteration for finding ``x`` such that ``g(x) = x``."""
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        x_next = g(x)
        history.append(x_next)
        if abs(x_next - x) < tol:
            return SolverResult(root=x_next, iterations=i, converged=True, history=history)
        x = x_next
    return SolverResult(root=x, iterations=max_iter, converged=False, history=history)
