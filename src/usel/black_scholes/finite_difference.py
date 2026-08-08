"""
Finite-difference solver for the Black-Scholes PDE.

This module numerically solves the Black-Scholes PDE

    dV/dt + (1/2) * sigma^2 * S^2 * V_SS + (r - q) * S * V_S - r * V = 0

on a uniform grid in the underlying price S, marching forward in
time-to-expiry tau = T - t from the terminal (payoff) condition at
tau = 0 to the price today at tau = T. This is the standard way to
price options when no closed-form solution exists (e.g. American
options, exotic payoffs, local/stochastic volatility) -- here it is
applied to the vanilla European case specifically so its output can
be validated against the exact closed-form price in `pricing.py`.

Discretization
--------------
Writing S_i = i * dS for i = 0, ..., N and using central differences
in S, the ds-dependence cancels exactly (a well-known property of
this particular discretization), leaving:

    dV_i/dtau = alpha_i * V_{i-1} + beta_i * V_i + gamma_i * V_{i+1}

    alpha_i = 0.5 * sigma^2 * i^2 - 0.5 * (r - q) * i
    beta_i  = -sigma^2 * i^2 - r
    gamma_i = 0.5 * sigma^2 * i^2 + 0.5 * (r - q) * i

A theta-method is used to step forward in tau:
    theta = 0.0  -> explicit (forward Euler); conditionally stable
    theta = 1.0  -> fully implicit (backward Euler); unconditionally
                    stable, first-order accurate in time
    theta = 0.5  -> Crank-Nicolson; unconditionally stable,
                    second-order accurate in time (default)

Boundary conditions at S = 0 and S = S_max are taken directly from
`pde.py` (`call_boundary_condition_lower/upper`,
`put_boundary_condition_lower/upper`), keeping the two modules
consistent by construction.

Implemented
-----------
- solve_black_scholes_pde()
- finite_difference_call_price()
- finite_difference_put_price()
- finite_difference_price()   (generic call/put dispatcher)

Future versions
----------------
- American-style early-exercise (projected SOR / penalty method)
- Non-uniform (e.g. log-price) grids for better accuracy near S=0
- Finite-difference Greeks read directly off the solved grid
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pde import (
    call_boundary_condition_lower,
    call_boundary_condition_upper,
    put_boundary_condition_lower,
    put_boundary_condition_upper,
    terminal_condition_call,
    terminal_condition_put,
)
from .pricing import (
    _asarray,
    _return_scalar_or_array,
    _validate_non_negative,
    _validate_positive,
)

_SCHEME_THETA = {
    "explicit": 0.0,
    "implicit": 1.0,
    "crank-nicolson": 0.5,
}

FiniteDifferenceGrid = namedtuple("FiniteDifferenceGrid", ["stock_grid", "price_grid"])


def _solve_tridiagonal(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    n = diag.shape[0]

    c_prime = np.empty(n, dtype=np.float64)
    d_prime = np.empty(n, dtype=np.float64)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x = np.empty(n, dtype=np.float64)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def _validate_scheme(scheme: str) -> float:
    scheme_normalized = scheme.strip().lower()
    if scheme_normalized not in _SCHEME_THETA:
        raise ValueError(f"scheme must be one of {sorted(_SCHEME_THETA)}, got {scheme!r}.")
    return _SCHEME_THETA[scheme_normalized]


def _validate_grid_params(
    num_space_steps: int,
    num_time_steps: int,
) -> None:
    if num_space_steps < 2:
        raise ValueError("num_space_steps must be at least 2.")
    if num_time_steps < 1:
        raise ValueError("num_time_steps must be at least 1.")


def solve_black_scholes_pde(
    option_type: str,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    stock_price_max: float | None = None,
    num_space_steps: int = 200,
    num_time_steps: int = 200,
    scheme: str = "crank-nicolson",
) -> FiniteDifferenceGrid:
    option_type_normalized = option_type.strip().lower()
    if option_type_normalized not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")

    theta = _validate_scheme(scheme)
    _validate_grid_params(num_space_steps, num_time_steps)

    strike_price_arr = _asarray(strike_price)
    volatility_arr = _asarray(volatility)
    time_to_expiry_arr = _asarray(time_to_expiry)

    _validate_positive(strike_price_arr, "strike_price")
    _validate_positive(volatility_arr, "volatility")
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")

    if stock_price_max is None:
        stock_price_max = 4.0 * float(strike_price)
    if stock_price_max <= 0.0:
        raise ValueError("stock_price_max must be greater than zero.")

    num_space_steps = int(num_space_steps)
    num_time_steps = int(num_time_steps)

    dS = stock_price_max / num_space_steps
    stock_grid = dS * np.arange(num_space_steps + 1, dtype=np.float64)

    if option_type_normalized == "call":
        price_grid = np.asarray(
            terminal_condition_call(stock_grid, strike_price),
            dtype=np.float64,
        )
    else:
        price_grid = np.asarray(
            terminal_condition_put(stock_grid, strike_price),
            dtype=np.float64,
        )

    if float(time_to_expiry) == 0.0:
        return FiniteDifferenceGrid(stock_grid, price_grid)

    dtau = float(time_to_expiry) / num_time_steps
    sigma2 = float(volatility) ** 2
    r = float(risk_free_rate)
    q = float(dividend_yield)

    node_index = np.arange(num_space_steps + 1, dtype=np.float64)
    alpha = 0.5 * sigma2 * node_index**2 - 0.5 * (r - q) * node_index
    beta = -sigma2 * node_index**2 - r
    gamma_coef = 0.5 * sigma2 * node_index**2 + 0.5 * (r - q) * node_index

    interior = slice(1, num_space_steps)

    if theta == 0.0:
        max_coefficient = sigma2 * num_space_steps**2 + abs(r)
        if dtau * max_coefficient > 1.0:
            warnings.warn(
                "explicit finite-difference scheme may be unstable for "
                "this grid (dtau too large relative to dS); consider "
                "increasing num_time_steps or switching to "
                "scheme='implicit' or scheme='crank-nicolson'.",
                RuntimeWarning,
                stacklevel=2,
            )

    for step in range(num_time_steps):
        tau_new = (step + 1) * dtau

        if option_type_normalized == "call":
            boundary_lower = call_boundary_condition_lower(tau_new)
            boundary_upper = call_boundary_condition_upper(
                stock_price_max, strike_price, risk_free_rate, tau_new, q
            )
        else:
            boundary_lower = put_boundary_condition_lower(strike_price, risk_free_rate, tau_new)
            boundary_upper = put_boundary_condition_upper(stock_price_max)

        old_price_grid = price_grid

        if theta == 0.0:
            new_interior = old_price_grid[interior] + dtau * (
                alpha[interior] * old_price_grid[0 : num_space_steps - 1]
                + beta[interior] * old_price_grid[interior]
                + gamma_coef[interior] * old_price_grid[2 : num_space_steps + 1]
            )
        else:
            lower = -theta * dtau * alpha[interior]
            diag = 1.0 - theta * dtau * beta[interior]
            upper = -theta * dtau * gamma_coef[interior]

            rhs = (
                (1.0 - theta) * dtau * alpha[interior] * old_price_grid[0 : num_space_steps - 1]
                + (1.0 + (1.0 - theta) * dtau * beta[interior]) * old_price_grid[interior]
                + (1.0 - theta)
                * dtau
                * gamma_coef[interior]
                * old_price_grid[2 : num_space_steps + 1]
            )

            rhs[0] += theta * dtau * alpha[1] * boundary_lower
            rhs[-1] += theta * dtau * gamma_coef[num_space_steps - 1] * boundary_upper

            new_interior = _solve_tridiagonal(lower, diag, upper, rhs)

        price_grid = np.empty_like(old_price_grid)
        price_grid[0] = boundary_lower
        price_grid[interior] = new_interior
        price_grid[-1] = boundary_upper

    return FiniteDifferenceGrid(stock_grid, price_grid)


def _interpolate_price(
    stock_price: ArrayLike,
    stock_grid: NDArray[np.float64],
    price_grid: NDArray[np.float64],
):
    stock_price_arr = _asarray(stock_price)

    if np.any(stock_price_arr < stock_grid[0]) or np.any(stock_price_arr > stock_grid[-1]):
        raise ValueError(
            "stock_price is outside the solved grid range "
            f"[{stock_grid[0]}, {stock_grid[-1]}]. Increase "
            "stock_price_max to cover the requested stock price."
        )

    result = np.interp(stock_price_arr, stock_grid, price_grid)

    return _return_scalar_or_array(stock_price, result)


def finite_difference_call_price(
    stock_price: ArrayLike,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    stock_price_max: float | None = None,
    num_space_steps: int = 200,
    num_time_steps: int = 200,
    scheme: str = "crank-nicolson",
):
    if stock_price_max is None:
        stock_price_arr = _asarray(stock_price)
        stock_price_max = 4.0 * max(float(np.max(stock_price_arr)), float(strike_price))

    stock_grid, price_grid = solve_black_scholes_pde(
        "call",
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
        stock_price_max,
        num_space_steps,
        num_time_steps,
        scheme,
    )

    return _interpolate_price(stock_price, stock_grid, price_grid)


def finite_difference_put_price(
    stock_price: ArrayLike,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    stock_price_max: float | None = None,
    num_space_steps: int = 200,
    num_time_steps: int = 200,
    scheme: str = "crank-nicolson",
):
    if stock_price_max is None:
        stock_price_arr = _asarray(stock_price)
        stock_price_max = 4.0 * max(float(np.max(stock_price_arr)), float(strike_price))

    stock_grid, price_grid = solve_black_scholes_pde(
        "put",
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
        stock_price_max,
        num_space_steps,
        num_time_steps,
        scheme,
    )

    return _interpolate_price(stock_price, stock_grid, price_grid)


def finite_difference_price(
    option_type: str,
    stock_price: ArrayLike,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    stock_price_max: float | None = None,
    num_space_steps: int = 200,
    num_time_steps: int = 200,
    scheme: str = "crank-nicolson",
):
    option_type_normalized = option_type.strip().lower()

    if option_type_normalized == "call":
        return finite_difference_call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
            stock_price_max,
            num_space_steps,
            num_time_steps,
            scheme,
        )
    elif option_type_normalized == "put":
        return finite_difference_put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
            stock_price_max,
            num_space_steps,
            num_time_steps,
            scheme,
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")
