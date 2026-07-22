"""Ordinary differential equation solvers: Euler, Improved Euler, RK2, RK4.

All solvers integrate first-order initial value problems of the form::

    dy/dt = f(t, y),  y(t0) = y0

over a fixed step size ``h`` from ``t0`` to ``t_end``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from usel.exceptions import ValidationError

ODEFunc = Callable[[float, float], float]


def _validate_step(t0: float, t_end: float, h: float) -> int:
    if h <= 0:
        raise ValidationError("Step size h must be positive")
    if t_end <= t0:
        raise ValidationError("t_end must be greater than t0")
    n_steps = int(round((t_end - t0) / h))
    return n_steps


def euler(f: ODEFunc, t0: float, y0: float, t_end: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Explicit (forward) Euler method. Returns (t_values, y_values)."""
    n_steps = _validate_step(t0, t_end, h)
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
    y = np.empty(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y


def improved_euler(
    f: ODEFunc, t0: float, y0: float, t_end: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Improved Euler (Heun's) method. Returns (t_values, y_values)."""
    n_steps = _validate_step(t0, t_end, h)
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
    y = np.empty(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        predictor = y[i] + h * k1
        k2 = f(t[i] + h, predictor)
        y[i + 1] = y[i] + (h / 2.0) * (k1 + k2)
    return t, y


def rk2(f: ODEFunc, t0: float, y0: float, t_end: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-order Runge-Kutta (midpoint) method. Returns (t_values, y_values)."""
    n_steps = _validate_step(t0, t_end, h)
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
    y = np.empty(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2.0, y[i] + (h / 2.0) * k1)
        y[i + 1] = y[i] + h * k2
    return t, y


def rk4(f: ODEFunc, t0: float, y0: float, t_end: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Classic fourth-order Runge-Kutta method. Returns (t_values, y_values)."""
    n_steps = _validate_step(t0, t_end, h)
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
    y = np.empty(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2.0, y[i] + (h / 2.0) * k1)
        k3 = f(t[i] + h / 2.0, y[i] + (h / 2.0) * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y
