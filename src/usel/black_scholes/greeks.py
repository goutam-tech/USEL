"""
Black-Scholes Greeks.

This module implements the first-order sensitivities ("Greeks") of
European option prices under the Black-Scholes model with respect to
the underlying price, volatility, time, and interest rate.

Implemented
-----------
- Delta (call, put)
- Gamma
- Vega
- Theta (call, put)
- Rho (call, put)

Future versions
----------------
- Vanna
- Vomma (Volga)
- Charm
- Speed
- Color
- Zomma
- Ultima
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .distributions import normal_cdf, normal_pdf
from .parameters import d1, d2
from .pricing import (
    _broadcast_inputs,
    _discount_strike,
    _return_scalar_or_array,
    _validate_inputs,
)


def _prepare(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike,
):
    (
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        volatility_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    ) = _broadcast_inputs(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    _validate_inputs(
        stock_price_arr,
        strike_price_arr,
        volatility_arr,
        time_to_expiry_arr,
    )

    return (
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        volatility_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    )


def delta_call(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S, K, r, sigma, T, q)
        bs_delta = np.exp(-q * T) * normal_cdf(d1_value)

    expiry_delta = np.where(S > K, 1.0, 0.0)
    result = np.where(T == 0.0, expiry_delta, bs_delta)

    return _return_scalar_or_array(stock_price, result)


def delta_put(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S, K, r, sigma, T, q)
        bs_delta = -np.exp(-q * T) * normal_cdf(-d1_value)

    expiry_delta = np.where(S < K, -1.0, 0.0)
    result = np.where(T == 0.0, expiry_delta, bs_delta)

    return _return_scalar_or_array(stock_price, result)


def gamma(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S, K, r, sigma, T, q)
        bs_gamma = np.exp(-q * T) * normal_pdf(d1_value) / (S * sigma * np.sqrt(T))

    result = np.where(T == 0.0, 0.0, bs_gamma)

    return _return_scalar_or_array(stock_price, result)


def vega(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S, K, r, sigma, T, q)
        bs_vega = S * np.exp(-q * T) * normal_pdf(d1_value) * np.sqrt(T)

    result = np.where(T == 0.0, 0.0, bs_vega)

    return _return_scalar_or_array(stock_price, result)


def theta_call(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S, K, r, sigma, T, q)
        d2_value = d2(S, K, r, sigma, T, q)
        discounted_strike = _discount_strike(K, r, T)

        bs_theta = (
            -S * np.exp(-q * T) * normal_pdf(d1_value) * sigma / (2.0 * np.sqrt(T))
            + q * S * np.exp(-q * T) * normal_cdf(d1_value)
            - r * discounted_strike * normal_cdf(d2_value)
        )

    result = np.where(T == 0.0, 0.0, bs_theta)

    return _return_scalar_or_array(stock_price, result)


def theta_put(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S, K, r, sigma, T, q)
        d2_value = d2(S, K, r, sigma, T, q)
        discounted_strike = _discount_strike(K, r, T)

        bs_theta = (
            -S * np.exp(-q * T) * normal_pdf(d1_value) * sigma / (2.0 * np.sqrt(T))
            - q * S * np.exp(-q * T) * normal_cdf(-d1_value)
            + r * discounted_strike * normal_cdf(-d2_value)
        )

    result = np.where(T == 0.0, 0.0, bs_theta)

    return _return_scalar_or_array(stock_price, result)


def rho_call(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d2_value = d2(S, K, r, sigma, T, q)
        discounted_strike = _discount_strike(K, r, T)
        bs_rho = discounted_strike * T * normal_cdf(d2_value)

    result = np.where(T == 0.0, 0.0, bs_rho)

    return _return_scalar_or_array(stock_price, result)


def rho_put(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    S, K, r, sigma, T, q = _prepare(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d2_value = d2(S, K, r, sigma, T, q)
        discounted_strike = _discount_strike(K, r, T)
        bs_rho = -discounted_strike * T * normal_cdf(-d2_value)

    result = np.where(T == 0.0, 0.0, bs_rho)

    return _return_scalar_or_array(stock_price, result)
