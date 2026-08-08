"""
Implied volatility for European options under the Black-Scholes model.

Given an observed market price, solves for the volatility (sigma) that
reproduces that price under the Black-Scholes formula.

Method
------
A safeguarded Newton-Raphson solver: at each iteration a Newton step is
attempted using vega as the derivative, but the iterate is always kept
inside a shrinking bisection bracket [low, high]. If the Newton step
would leave the bracket, is non-finite, or vega is too small to trust
(near-zero vega, e.g. deep ITM/OTM or near expiry), a bisection step is
used instead. This guarantees convergence (like bisection) with
near-quadratic speed in well-behaved regions (like Newton-Raphson).

Convergence requires both the price residual and the bisection bracket
width to be small. Price alone is not sufficient: for deep-OTM or
near-expiry options, price and vega can both be vanishingly small, so a
wide range of volatilities can satisfy an absolute price tolerance long
before the bracket has actually narrowed to the true root.

Implemented
-----------
- implied_volatility_call()
- implied_volatility_put()
- implied_volatility()   (generic call/put dispatcher)

Future versions
----------------
- Vectorized Jaeckel "Let's Be Rational" closed-form initial guess
- American option implied volatility (requires PDE/binomial pricer)
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import ImpliedConstants
from .greeks import vega as _vega
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


def _validate_market_price_call(
    market_price: NDArray[np.float64],
    stock_price: NDArray[np.float64],
    strike_price: NDArray[np.float64],
    risk_free_rate: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
    dividend_yield: NDArray[np.float64],
) -> None:
    discounted_stock = _discount_stock(stock_price, dividend_yield, time_to_expiry)
    discounted_strike = _discount_strike(strike_price, risk_free_rate, time_to_expiry)
    intrinsic = np.maximum(discounted_stock - discounted_strike, 0.0)

    below = market_price < (intrinsic - ImpliedConstants.ARBITRAGE_TOL)
    above = market_price > (discounted_stock + ImpliedConstants.ARBITRAGE_TOL)

    if np.any(below) or np.any(above):
        raise ValueError(
            "market_price violates no-arbitrage bounds for a European "
            "call: price must satisfy max(S*e^(-qT) - K*e^(-rT), 0) "
            "<= price <= S*e^(-qT)."
        )


def _validate_market_price_put(
    market_price: NDArray[np.float64],
    stock_price: NDArray[np.float64],
    strike_price: NDArray[np.float64],
    risk_free_rate: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
    dividend_yield: NDArray[np.float64],
) -> None:
    discounted_stock = _discount_stock(stock_price, dividend_yield, time_to_expiry)
    discounted_strike = _discount_strike(strike_price, risk_free_rate, time_to_expiry)
    intrinsic = np.maximum(discounted_strike - discounted_stock, 0.0)

    below = market_price < (intrinsic - ImpliedConstants.ARBITRAGE_TOL)
    above = market_price > (discounted_strike + ImpliedConstants.ARBITRAGE_TOL)

    if np.any(below) or np.any(above):
        raise ValueError(
            "market_price violates no-arbitrage bounds for a European "
            "put: price must satisfy max(K*e^(-rT) - S*e^(-qT), 0) "
            "<= price <= K*e^(-rT)."
        )


def _solve_implied_volatility(
    price_fn,
    market_price: NDArray[np.float64],
    stock_price: NDArray[np.float64],
    strike_price: NDArray[np.float64],
    risk_free_rate: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
    dividend_yield: NDArray[np.float64],
    initial_guess: NDArray[np.float64],
    low_vol: float,
    high_vol: float,
    tol: float,
    max_iterations: int,
) -> NDArray[np.float64]:
    low = np.full_like(stock_price, low_vol, dtype=np.float64)
    high = np.full_like(stock_price, high_vol, dtype=np.float64)
    vol = np.clip(initial_guess, low_vol, high_vol).astype(np.float64)

    converged = np.zeros_like(stock_price, dtype=bool)

    bracket_tol = 1e-10

    for _ in range(max_iterations):
        price = price_fn(
            stock_price,
            strike_price,
            risk_free_rate,
            vol,
            time_to_expiry,
            dividend_yield,
        )
        diff = price - market_price

        newly_converged = (np.abs(diff) < tol) & ((high - low) < bracket_tol)
        converged = converged | newly_converged

        if np.all(converged):
            break

        price_too_high = diff > 0.0
        high = np.where(price_too_high, vol, high)
        low = np.where(~price_too_high, vol, low)

        with np.errstate(divide="ignore", invalid="ignore"):
            v = _vega(
                stock_price,
                strike_price,
                risk_free_rate,
                vol,
                time_to_expiry,
                dividend_yield,
            )
            newton_vol = vol - diff / v

        newton_in_bracket = (
            np.isfinite(newton_vol) & (newton_vol > low) & (newton_vol < high) & (v > 1e-12)
        )

        bisection_vol = 0.5 * (low + high)

        next_vol = np.where(newton_in_bracket, newton_vol, bisection_vol)

        vol = np.where(converged, vol, next_vol)

    if not np.all(converged):
        n_failed = int(np.sum(~converged))
        warnings.warn(
            f"implied volatility solver did not converge for "
            f"{n_failed} element(s) within {max_iterations} "
            f"iterations; returning best available estimate.",
            RuntimeWarning,
            stacklevel=3,
        )

    return vol


def implied_volatility_call(
    market_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    initial_guess: ArrayLike = ImpliedConstants.DEFAULT_INITIAL_GUESS,
    low_vol: float = ImpliedConstants.DEFAULT_LOW_VOL,
    high_vol: float = ImpliedConstants.DEFAULT_HIGH_VOL,
    tol: float = ImpliedConstants.DEFAULT_TOLERANCE,
    max_iterations: int = ImpliedConstants.DEFAULT_MAX_ITERATIONS,
):
    (
        market_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    ) = np.broadcast_arrays(
        _asarray(market_price),
        _asarray(stock_price),
        _asarray(strike_price),
        _asarray(risk_free_rate),
        _asarray(time_to_expiry),
        _asarray(dividend_yield),
    )

    _validate_positive(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")
    _validate_positive(time_to_expiry_arr, "time_to_expiry")
    _validate_non_negative(market_price_arr, "market_price")

    _validate_market_price_call(
        market_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    )

    initial_guess_arr = np.broadcast_to(_asarray(initial_guess), stock_price_arr.shape).astype(
        np.float64
    )

    result = _solve_implied_volatility(
        _call_price,
        market_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
        initial_guess_arr,
        low_vol,
        high_vol,
        tol,
        max_iterations,
    )

    return _return_scalar_or_array(market_price, result)


def implied_volatility_put(
    market_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    initial_guess: ArrayLike = ImpliedConstants.DEFAULT_INITIAL_GUESS,
    low_vol: float = ImpliedConstants.DEFAULT_LOW_VOL,
    high_vol: float = ImpliedConstants.DEFAULT_HIGH_VOL,
    tol: float = ImpliedConstants.DEFAULT_TOLERANCE,
    max_iterations: int = ImpliedConstants.DEFAULT_MAX_ITERATIONS,
):
    (
        market_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    ) = np.broadcast_arrays(
        _asarray(market_price),
        _asarray(stock_price),
        _asarray(strike_price),
        _asarray(risk_free_rate),
        _asarray(time_to_expiry),
        _asarray(dividend_yield),
    )

    _validate_positive(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")
    _validate_positive(time_to_expiry_arr, "time_to_expiry")
    _validate_non_negative(market_price_arr, "market_price")

    _validate_market_price_put(
        market_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    )

    initial_guess_arr = np.broadcast_to(_asarray(initial_guess), stock_price_arr.shape).astype(
        np.float64
    )

    result = _solve_implied_volatility(
        _put_price,
        market_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
        initial_guess_arr,
        low_vol,
        high_vol,
        tol,
        max_iterations,
    )

    return _return_scalar_or_array(market_price, result)


def implied_volatility(
    market_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    option_type: str = "call",
    dividend_yield: ArrayLike = 0.0,
    initial_guess: ArrayLike = ImpliedConstants.DEFAULT_INITIAL_GUESS,
    low_vol: float = ImpliedConstants.DEFAULT_LOW_VOL,
    high_vol: float = ImpliedConstants.DEFAULT_HIGH_VOL,
    tol: float = ImpliedConstants.DEFAULT_TOLERANCE,
    max_iterations: int = ImpliedConstants.DEFAULT_MAX_ITERATIONS,
):
    option_type_normalized = option_type.strip().lower()

    if option_type_normalized == "call":
        return implied_volatility_call(
            market_price,
            stock_price,
            strike_price,
            risk_free_rate,
            time_to_expiry,
            dividend_yield,
            initial_guess,
            low_vol,
            high_vol,
            tol,
            max_iterations,
        )
    elif option_type_normalized == "put":
        return implied_volatility_put(
            market_price,
            stock_price,
            strike_price,
            risk_free_rate,
            time_to_expiry,
            dividend_yield,
            initial_guess,
            low_vol,
            high_vol,
            tol,
            max_iterations,
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")
