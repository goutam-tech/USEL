"""
Black-Scholes PDE formulation.

This module implements the analytical building blocks of the
Black-Scholes partial differential equation: the PDE residual itself,
terminal (payoff) conditions, and boundary conditions at S -> 0 and
S -> infinity.

Black-Scholes PDE
------------------
    dV/dt + (1/2) * sigma^2 * S^2 * V_SS + (r - q) * S * V_S - r * V = 0

where V(S, t) is the option value, t is calendar time, and the
subscripts denote partial derivatives with respect to S.

Because `pricing.call_price` / `pricing.put_price` are parametrized by
*time to expiry* T rather than calendar time t, and t = t_expiry - T,
we have dV/dt = -dV/dT. The `greeks.theta_call` / `greeks.theta_put`
functions already implement time decay in this calendar-time
convention, so they plug directly into the PDE below with no sign
correction needed.

This module verifies -- rather than numerically solves -- the PDE: it
uses the closed-form price (`pricing`) together with the closed-form
Greeks (`greeks`) to compute an exact PDE residual analytically. A
grid-based numerical solver belongs in `finite_difference.py`.

Implemented
-----------
- black_scholes_pde_residual()
- verify_black_scholes_pde()
- terminal_condition_call() / terminal_condition_put()
- call_boundary_condition_lower() / call_boundary_condition_upper()
- put_boundary_condition_lower() / put_boundary_condition_upper()

Future versions
----------------
- Finite-difference PDE solver (see finite_difference.py)
- American-style early-exercise boundary / free-boundary conditions
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .constants import PdeConstants
from .greeks import delta_call, delta_put, gamma, theta_call, theta_put
from .pricing import (
    _asarray,
    _discount_stock,
    _discount_strike,
    _return_scalar_or_array,
    _validate_non_negative,
    _validate_positive,
)
from .pricing import call_price as _call_price
from .pricing import put_price as _put_price


def black_scholes_pde_residual(
    option_type: str,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    option_type_normalized = option_type.strip().lower()

    if option_type_normalized == "call":
        value = _call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        value_t = theta_call(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        value_s = delta_call(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
    elif option_type_normalized == "put":
        value = _put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        value_t = theta_put(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        value_s = delta_put(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")

    value_ss = gamma(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    stock_price_arr = _asarray(stock_price)
    risk_free_rate_arr = _asarray(risk_free_rate)
    volatility_arr = _asarray(volatility)
    dividend_yield_arr = _asarray(dividend_yield)
    value_arr = _asarray(value)
    value_t_arr = _asarray(value_t)
    value_s_arr = _asarray(value_s)
    value_ss_arr = _asarray(value_ss)

    residual = (
        value_t_arr
        + 0.5 * volatility_arr**2 * stock_price_arr**2 * value_ss_arr
        + (risk_free_rate_arr - dividend_yield_arr) * stock_price_arr * value_s_arr
        - risk_free_rate_arr * value_arr
    )

    return _return_scalar_or_array(stock_price, residual)


def verify_black_scholes_pde(
    option_type: str,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    tol: float = PdeConstants.DEFAULT_PDE_TOLERANCE,
):
    residual = black_scholes_pde_residual(
        option_type,
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    residual_arr = _asarray(residual)
    satisfied = bool(np.all(np.abs(residual_arr) <= tol))

    return residual, satisfied


def terminal_condition_call(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
):
    stock_price_arr = _asarray(stock_price)
    strike_price_arr = _asarray(strike_price)

    _validate_non_negative(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")

    payoff = np.maximum(stock_price_arr - strike_price_arr, 0.0)

    return _return_scalar_or_array(stock_price, payoff)


def terminal_condition_put(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
):
    stock_price_arr = _asarray(stock_price)
    strike_price_arr = _asarray(strike_price)

    _validate_non_negative(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")

    payoff = np.maximum(strike_price_arr - stock_price_arr, 0.0)

    return _return_scalar_or_array(stock_price, payoff)


def call_boundary_condition_lower(
    time_to_expiry: ArrayLike,
):
    time_to_expiry_arr = _asarray(time_to_expiry)
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")

    result = np.zeros_like(time_to_expiry_arr)

    return _return_scalar_or_array(time_to_expiry, result)


def call_boundary_condition_upper(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    stock_price_arr = _asarray(stock_price)
    strike_price_arr = _asarray(strike_price)
    risk_free_rate_arr = _asarray(risk_free_rate)
    time_to_expiry_arr = _asarray(time_to_expiry)
    dividend_yield_arr = _asarray(dividend_yield)

    _validate_positive(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")

    discounted_stock = _discount_stock(stock_price_arr, dividend_yield_arr, time_to_expiry_arr)
    discounted_strike = _discount_strike(strike_price_arr, risk_free_rate_arr, time_to_expiry_arr)

    result = discounted_stock - discounted_strike

    return _return_scalar_or_array(stock_price, result)


def put_boundary_condition_lower(
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
):
    strike_price_arr = _asarray(strike_price)
    risk_free_rate_arr = _asarray(risk_free_rate)
    time_to_expiry_arr = _asarray(time_to_expiry)

    _validate_positive(strike_price_arr, "strike_price")
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")

    result = _discount_strike(strike_price_arr, risk_free_rate_arr, time_to_expiry_arr)

    return _return_scalar_or_array(strike_price, result)


def put_boundary_condition_upper(
    stock_price: ArrayLike,
):
    stock_price_arr = _asarray(stock_price)
    _validate_positive(stock_price_arr, "stock_price")

    result = np.zeros_like(stock_price_arr)

    return _return_scalar_or_array(stock_price, result)
