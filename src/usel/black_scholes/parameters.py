from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _asarray(value: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _return_scalar_or_array(
    original: ArrayLike,
    result: NDArray[np.float64],
):
    if np.isscalar(original):
        return float(result)

    return result


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


def d1(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    original = stock_price

    stock_price = _asarray(stock_price)
    strike_price = _asarray(strike_price)
    risk_free_rate = _asarray(risk_free_rate)
    volatility = _asarray(volatility)
    time_to_expiry = _asarray(time_to_expiry)
    dividend_yield = _asarray(dividend_yield)

    _validate_inputs(
        stock_price,
        strike_price,
        volatility,
        time_to_expiry,
    )

    result = np.empty_like(stock_price, dtype=np.float64)

    zero_mask = time_to_expiry == 0.0

    if np.any(zero_mask):
        result[zero_mask] = np.where(
            stock_price[zero_mask] > strike_price[zero_mask],
            np.inf,
            np.where(
                stock_price[zero_mask] < strike_price[zero_mask],
                -np.inf,
                0.0,
            ),
        )

    positive_mask = ~zero_mask

    if np.any(positive_mask):
        s = stock_price[positive_mask]
        k = strike_price[positive_mask]
        r = risk_free_rate[positive_mask]
        sigma = volatility[positive_mask]
        t = time_to_expiry[positive_mask]
        q = dividend_yield[positive_mask]

        numerator = np.log(s / k) + (r - q + 0.5 * sigma**2) * t

        denominator = sigma * np.sqrt(t)

        result[positive_mask] = numerator / denominator

    return _return_scalar_or_array(original, result)


def d2(
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    original = stock_price

    stock_price = _asarray(stock_price)
    strike_price = _asarray(strike_price)
    risk_free_rate = _asarray(risk_free_rate)
    volatility = _asarray(volatility)
    time_to_expiry = _asarray(time_to_expiry)
    dividend_yield = _asarray(dividend_yield)

    d1_value = d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        dividend_yield=dividend_yield,
    )

    d1_value = _asarray(d1_value)

    result = np.empty_like(d1_value)

    zero_mask = time_to_expiry == 0.0

    if np.any(zero_mask):
        result[zero_mask] = d1_value[zero_mask]

    positive_mask = ~zero_mask

    if np.any(positive_mask):
        result[positive_mask] = d1_value[positive_mask] - volatility[positive_mask] * np.sqrt(
            time_to_expiry[positive_mask]
        )

    return _return_scalar_or_array(original, result)
