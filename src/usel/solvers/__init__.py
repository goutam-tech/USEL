"""Numerical solvers: root finding, ODE integration, and quadrature."""

from __future__ import annotations

from usel.solvers.integration import gaussian_quadrature, simpson, trapezoidal
from usel.solvers.ode import euler, improved_euler, rk2, rk4
from usel.solvers.root_finding import bisection, fixed_point, gradient_descent, newton, secant

__all__ = [
    "newton",
    "bisection",
    "secant",
    "gradient_descent",
    "fixed_point",
    "euler",
    "improved_euler",
    "rk2",
    "rk4",
    "trapezoidal",
    "simpson",
    "gaussian_quadrature",
]
