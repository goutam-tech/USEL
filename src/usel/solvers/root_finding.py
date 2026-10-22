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
    """Result of a root-finding or optimization routine.

    Attributes:
        root: The approximate root or minimizer.
        iterations: Number of iterations performed before termination.
        converged: ``True`` when the method met its tolerance before
            ``max_iter``; ``False`` when the iteration limit was reached.
        history: Sequence of iterates, starting with the initial guess.
        residual: Absolute value of the defining equation evaluated at
            ``root`` (``|f(root)|`` for root finding, ``|df(root)|`` for
            gradient descent, ``|g(root) - root|`` for fixed-point
            iteration). ``None`` when not computed.

    Termination status:
        * ``converged=True`` when the method met its tolerance.
        * ``converged=False`` when ``max_iter`` was reached without meeting
          the tolerance (iteration limit).
        * Invalid input or numerical breakdown raises ``SolverError`` before
          returning a result.
    """

    root: float
    iterations: int
    converged: bool
    history: list[float]
    residual: float | None = None


def newton(
    f: ScalarFunc,
    df: ScalarFunc,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> SolverResult:
    """Newton-Raphson root finding for ``f(x) = 0`` given its derivative ``df``.

    Termination status:
        * ``converged=True`` when successive iterates agree within ``tol``.
        * ``converged=False`` when ``max_iter`` is reached without meeting
          ``tol``.
        * ``SolverError`` is raised when the derivative is zero (numerical
          breakdown).
    """
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
            return SolverResult(
                root=x_next,
                iterations=i,
                converged=True,
                history=history,
                residual=abs(f(x_next)),
            )
        x = x_next
    return SolverResult(
        root=x,
        iterations=max_iter,
        converged=False,
        history=history,
        residual=abs(f(x)),
    )


def bisection(
    f: ScalarFunc,
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> SolverResult:
    """Bisection method for root finding on a bracketing interval [a, b].

    Termination status:
        * ``converged=True`` when ``|f(midpoint)|`` or the half-bracket
          width falls within ``tol``.
        * ``converged=False`` when ``max_iter`` is reached.
        * ``SolverError`` is raised when ``f(a)`` and ``f(b)`` have the same
          sign (no bracket).
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise SolverError("f(a) and f(b) must have opposite signs for bisection")

    history = []
    for i in range(1, max_iter + 1):
        midpoint = (a + b) / 2.0
        f_mid = f(midpoint)
        history.append(midpoint)
        if abs(f_mid) < tol or (b - a) / 2.0 < tol:
            return SolverResult(
                root=midpoint,
                iterations=i,
                converged=True,
                history=history,
                residual=abs(f_mid),
            )
        if fa * f_mid < 0:
            b, fb = midpoint, f_mid
        else:
            a, fa = midpoint, f_mid
    root = (a + b) / 2.0
    return SolverResult(
        root=root,
        iterations=max_iter,
        converged=False,
        history=history,
        residual=abs(f(root)),
    )


def secant(
    f: ScalarFunc,
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> SolverResult:
    """Secant method for root finding using two initial guesses.

    Termination status:
        * ``converged=True`` when successive iterates agree within ``tol``.
        * ``converged=False`` when ``max_iter`` is reached.
        * ``SolverError`` is raised when the denominator ``f(x1) - f(x0)``
          is zero (numerical breakdown).
    """
    history = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for i in range(1, max_iter + 1):
        if f1 - f0 == 0:
            raise SolverError("Zero denominator encountered in secant method")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            return SolverResult(
                root=x2,
                iterations=i,
                converged=True,
                history=history,
                residual=abs(f(x2)),
            )
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    return SolverResult(
        root=x1,
        iterations=max_iter,
        converged=False,
        history=history,
        residual=abs(f1),
    )


def gradient_descent(
    df: ScalarFunc,
    x0: float,
    learning_rate: float = 0.01,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> SolverResult:
    """Gradient descent minimization given the derivative ``df`` of a 1D function.

    The "root" is the minimizer where ``df(x) = 0``.

    Termination status:
        * ``converged=True`` when successive iterates agree within ``tol``.
        * ``converged=False`` when ``max_iter`` is reached.
    """
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        grad = df(x)
        x_next = x - learning_rate * grad
        history.append(x_next)
        if abs(x_next - x) < tol:
            return SolverResult(
                root=x_next,
                iterations=i,
                converged=True,
                history=history,
                residual=abs(df(x_next)),
            )
        x = x_next
    return SolverResult(
        root=x,
        iterations=max_iter,
        converged=False,
        history=history,
        residual=abs(df(x)),
    )


def fixed_point(
    g: ScalarFunc,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult:
    """Fixed-point iteration for finding ``x`` such that ``g(x) = x``.

    Termination status:
        * ``converged=True`` when successive iterates agree within ``tol``.
        * ``converged=False`` when ``max_iter`` is reached.
    """
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        x_next = g(x)
        history.append(x_next)
        if abs(x_next - x) < tol:
            return SolverResult(
                root=x_next,
                iterations=i,
                converged=True,
                history=history,
                residual=abs(g(x_next) - x_next),
            )
        x = x_next
    return SolverResult(
        root=x,
        iterations=max_iter,
        converged=False,
        history=history,
        residual=abs(g(x) - x),
    )
