"""
Monte Carlo (stochastic) pricing for European options under the
Black-Scholes model.

This module prices options by simulating terminal (and, optionally,
full-path) trajectories of Geometric Brownian Motion (GBM) under the
risk-neutral measure, then averaging discounted payoffs. It serves as
a third, independent validation path alongside the closed-form
solution (`pricing.py`) and the finite-difference PDE solver
(`finite_difference.py`) -- all three should agree, within their
respective error bars, for any vanilla European option.

Risk-neutral GBM
-----------------
Under the risk-neutral measure, the terminal stock price satisfies:

    S_T = S_0 * exp( (r - q - sigma^2/2) * T + sigma * sqrt(T) * Z )

where Z ~ N(0, 1). This module draws Z with NumPy's `default_rng`
Generator and evaluates the payoff at S_T (for vanilla options, only
the terminal price is needed; `simulate_gbm_paths` additionally
supports full intermediate-step paths for future path-dependent /
exotic payoffs).

Variance reduction
-------------------
Antithetic variates are used by default: each draw Z is paired with
its mirror image -Z, which reduces the Monte Carlo standard error for
roughly the same computational cost (exact reduction factor depends
on the payoff's convexity).

Implemented
-----------
- simulate_gbm_terminal()
- simulate_gbm_paths()
- monte_carlo_call_price()
- monte_carlo_put_price()
- monte_carlo_price()   (generic call/put dispatcher)

Future versions
----------------
- Path-dependent payoffs (Asian, barrier, lookback options)
- Pathwise / likelihood-ratio Monte Carlo Greeks
- Control variates using the closed-form price as the control
- Quasi-Monte Carlo (Sobol) sampling for faster convergence

Note on inputs
---------------
Unlike `pricing.py` / `greeks.py`, this module accepts scalar inputs
only (not NumPy arrays) for `stock_price`, `strike_price`, etc. --
Monte Carlo simulation of a single scenario is already
computationally heavier than the closed-form/finite-difference
routes, and vectorizing over both simulation paths and multiple
market scenarios simultaneously would significantly complicate the
API for little practical benefit in this library.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from numpy.typing import NDArray

from .distributions import inverse_normal_cdf
from .pricing import _asarray, _validate_non_negative, _validate_positive

MonteCarloResult = namedtuple(
    "MonteCarloResult",
    ["price", "std_error", "conf_interval_lower", "conf_interval_upper", "num_paths"],
)


def _validate_scalar_positive(value: float, name: str) -> float:
    value = float(value)
    _validate_positive(_asarray(value), name)
    return value


def _validate_scalar_non_negative(value: float, name: str) -> float:
    value = float(value)
    _validate_non_negative(_asarray(value), name)
    return value


def _validate_num_paths(num_paths: int) -> int:
    num_paths = int(num_paths)
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2.")
    return num_paths


def _validate_confidence_level(confidence_level: float) -> float:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    return confidence_level


def simulate_gbm_terminal(
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    num_paths: int = 100_000,
    antithetic: bool = True,
    random_seed: int | None = None,
) -> NDArray[np.float64]:
    stock_price = _validate_scalar_positive(stock_price, "stock_price")
    volatility = _validate_scalar_positive(volatility, "volatility")
    time_to_expiry = _validate_scalar_non_negative(time_to_expiry, "time_to_expiry")
    risk_free_rate = float(risk_free_rate)
    dividend_yield = float(dividend_yield)
    num_paths = _validate_num_paths(num_paths)

    rng = np.random.default_rng(random_seed)

    if antithetic:
        half = num_paths // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])
        if num_paths % 2 == 1:
            z = np.concatenate([z, rng.standard_normal(1)])
    else:
        z = rng.standard_normal(num_paths)

    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    diffusion = volatility * np.sqrt(time_to_expiry) * z

    terminal_prices = stock_price * np.exp(drift + diffusion)

    return terminal_prices


def simulate_gbm_paths(
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    num_paths: int = 10_000,
    num_steps: int = 252,
    antithetic: bool = True,
    random_seed: int | None = None,
) -> NDArray[np.float64]:
    stock_price = _validate_scalar_positive(stock_price, "stock_price")
    volatility = _validate_scalar_positive(volatility, "volatility")
    time_to_expiry = _validate_scalar_non_negative(time_to_expiry, "time_to_expiry")
    risk_free_rate = float(risk_free_rate)
    dividend_yield = float(dividend_yield)
    num_paths = _validate_num_paths(num_paths)
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")

    dt = time_to_expiry / num_steps

    rng = np.random.default_rng(random_seed)

    if antithetic:
        half = num_paths // 2
        z_half = rng.standard_normal((half, num_steps))
        z = np.concatenate([z_half, -z_half], axis=0)
        if num_paths % 2 == 1:
            z = np.concatenate([z, rng.standard_normal((1, num_steps))], axis=0)
    else:
        z = rng.standard_normal((num_paths, num_steps))

    increments = (
        risk_free_rate - dividend_yield - 0.5 * volatility**2
    ) * dt + volatility * np.sqrt(dt) * z

    log_relative_paths = np.cumsum(increments, axis=1)
    log_relative_paths = np.hstack([np.zeros((num_paths, 1)), log_relative_paths])

    paths = stock_price * np.exp(log_relative_paths)

    return paths


def _summarize_discounted_payoff(
    discounted_payoff: NDArray[np.float64],
    confidence_level: float,
    reported_num_paths=None,
) -> MonteCarloResult:
    num_paths = reported_num_paths if reported_num_paths is not None else discounted_payoff.shape[0]

    price = float(np.mean(discounted_payoff))
    std_error = float(np.std(discounted_payoff, ddof=1) / np.sqrt(num_paths))

    tail_probability = (1.0 - confidence_level) / 2.0
    z_score = float(inverse_normal_cdf(1.0 - tail_probability))

    lower = price - z_score * std_error
    upper = price + z_score * std_error

    return MonteCarloResult(price, std_error, lower, upper, num_paths)


def monte_carlo_call_price(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    num_paths: int = 100_000,
    antithetic: bool = True,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> MonteCarloResult:
    stock_price = _validate_scalar_positive(stock_price, "stock_price")
    strike_price = _validate_scalar_positive(strike_price, "strike_price")
    time_to_expiry = _validate_scalar_non_negative(time_to_expiry, "time_to_expiry")
    risk_free_rate = float(risk_free_rate)
    confidence_level = _validate_confidence_level(confidence_level)

    terminal_prices = simulate_gbm_terminal(
        stock_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
        num_paths,
        antithetic,
        random_seed,
    )

    payoff = np.maximum(
        terminal_prices - strike_price,
        0.0,
    )

    if antithetic:
        pair_count = len(payoff) // 2

        if pair_count > 0:
            paired_payoff = (
                payoff[: pair_count * 2]
                .reshape(
                    pair_count,
                    2,
                )
                .mean(axis=1)
            )

            if len(payoff) % 2 == 1:
                payoff = np.concatenate(
                    [
                        paired_payoff,
                        payoff[-1:],
                    ]
                )
            else:
                payoff = paired_payoff

    discounted_payoff = np.exp(-risk_free_rate * time_to_expiry) * payoff

    return _summarize_discounted_payoff(
        discounted_payoff,
        confidence_level,
        num_paths,
    )


def monte_carlo_put_price(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    num_paths: int = 100_000,
    antithetic: bool = True,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> MonteCarloResult:
    stock_price = _validate_scalar_positive(stock_price, "stock_price")
    strike_price = _validate_scalar_positive(strike_price, "strike_price")
    time_to_expiry = _validate_scalar_non_negative(time_to_expiry, "time_to_expiry")
    risk_free_rate = float(risk_free_rate)
    confidence_level = _validate_confidence_level(confidence_level)

    terminal_prices = simulate_gbm_terminal(
        stock_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
        num_paths,
        antithetic,
        random_seed,
    )

    payoff = np.maximum(
        strike_price - terminal_prices,
        0.0,
    )

    if antithetic:
        pair_count = len(payoff) // 2

        paired_payoff = (
            payoff[: pair_count * 2]
            .reshape(
                pair_count,
                2,
            )
            .mean(axis=1)
        )

        if len(payoff) % 2 == 1:
            payoff = np.concatenate(
                [
                    paired_payoff,
                    payoff[-1:],
                ]
            )
        else:
            payoff = paired_payoff

    discounted_payoff = np.exp(-risk_free_rate * time_to_expiry) * payoff

    return _summarize_discounted_payoff(
        discounted_payoff,
        confidence_level,
        num_paths,
    )


def monte_carlo_price(
    option_type: str,
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    num_paths: int = 100_000,
    antithetic: bool = True,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> MonteCarloResult:
    option_type_normalized = option_type.strip().lower()

    if option_type_normalized == "call":
        return monte_carlo_call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
            num_paths,
            antithetic,
            confidence_level,
            random_seed,
        )
    elif option_type_normalized == "put":
        return monte_carlo_put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
            num_paths,
            antithetic,
            confidence_level,
            random_seed,
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")
