from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .distributions import normal_cdf
from .formulas import d1, d2


def _asarray(value: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _return_scalar_or_array(
    _original: ArrayLike,
    result: NDArray[np.float64],
):
    result_arr = np.asarray(result)

    if result_arr.ndim == 0:
        return float(result_arr)

    return result_arr


def _broadcast_inputs(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike,
):
    return np.broadcast_arrays(
        _asarray(stock_price),
        _asarray(strike_price),
        _asarray(risk_free_rate),
        _asarray(volatility),
        _asarray(time_to_expiry),
        _asarray(dividend_yield),
    )


def _validate_positive(
    value: NDArray[np.float64],
    name: str,
) -> None:
    if np.any(value <= 0.0):
        raise ValueError(f"{name} must be greater than zero.")


def _validate_non_negative(
    value: NDArray[np.float64],
    name: str,
) -> None:
    if np.any(value < 0.0):
        raise ValueError(f"{name} cannot be negative.")


def _validate_inputs(
    stock_price: NDArray[np.float64],
    strike_price: NDArray[np.float64],
    volatility: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
) -> None:
    _validate_positive(stock_price, "stock_price")
    _validate_positive(strike_price, "strike_price")
    _validate_positive(volatility, "volatility")
    _validate_non_negative(time_to_expiry, "time_to_expiry")


def _discount_stock(
    stock_price: NDArray[np.float64],
    dividend_yield: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
) -> NDArray[np.float64]:
    return stock_price * np.exp(-dividend_yield * time_to_expiry)


def _discount_strike(
    strike_price: NDArray[np.float64],
    risk_free_rate: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
) -> NDArray[np.float64]:
    return strike_price * np.exp(-risk_free_rate * time_to_expiry)


def call_price(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
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

    intrinsic_value = np.maximum(stock_price_arr - strike_price_arr, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(
            stock_price_arr,
            strike_price_arr,
            risk_free_rate_arr,
            volatility_arr,
            time_to_expiry_arr,
            dividend_yield_arr,
        )
        d2_value = d2(
            stock_price_arr,
            strike_price_arr,
            risk_free_rate_arr,
            volatility_arr,
            time_to_expiry_arr,
            dividend_yield_arr,
        )

        discounted_stock = _discount_stock(
            stock_price_arr,
            dividend_yield_arr,
            time_to_expiry_arr,
        )
        discounted_strike = _discount_strike(
            strike_price_arr,
            risk_free_rate_arr,
            time_to_expiry_arr,
        )

        black_scholes_price = discounted_stock * normal_cdf(
            d1_value
        ) - discounted_strike * normal_cdf(d2_value)

    result = np.where(
        time_to_expiry_arr == 0.0,
        intrinsic_value,
        black_scholes_price,
    )

    return _return_scalar_or_array(stock_price, result)


def put_price(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
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

    intrinsic_value = np.maximum(strike_price_arr - stock_price_arr, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(
            stock_price_arr,
            strike_price_arr,
            risk_free_rate_arr,
            volatility_arr,
            time_to_expiry_arr,
            dividend_yield_arr,
        )
        d2_value = d2(
            stock_price_arr,
            strike_price_arr,
            risk_free_rate_arr,
            volatility_arr,
            time_to_expiry_arr,
            dividend_yield_arr,
        )

        discounted_stock = _discount_stock(
            stock_price_arr,
            dividend_yield_arr,
            time_to_expiry_arr,
        )
        discounted_strike = _discount_strike(
            strike_price_arr,
            risk_free_rate_arr,
            time_to_expiry_arr,
        )

        black_scholes_price = discounted_strike * normal_cdf(
            -d2_value
        ) - discounted_stock * normal_cdf(-d1_value)

    result = np.where(
        time_to_expiry_arr == 0.0,
        intrinsic_value,
        black_scholes_price,
    )

    return _return_scalar_or_array(stock_price, result)
