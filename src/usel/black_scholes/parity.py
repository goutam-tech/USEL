from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .pricing import (
    _asarray,
    _discount_stock,
    _discount_strike,
    _return_scalar_or_array,
    _validate_non_negative,
    _validate_positive,
)


def call_from_put(
    put_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    (
        put_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    ) = np.broadcast_arrays(
        _asarray(put_price),
        _asarray(stock_price),
        _asarray(strike_price),
        _asarray(risk_free_rate),
        _asarray(time_to_expiry),
        _asarray(dividend_yield),
    )

    _validate_positive(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")
    _validate_non_negative(put_price_arr, "put_price")

    discounted_stock = _discount_stock(stock_price_arr, dividend_yield_arr, time_to_expiry_arr)
    discounted_strike = _discount_strike(strike_price_arr, risk_free_rate_arr, time_to_expiry_arr)

    call = put_price_arr + discounted_stock - discounted_strike

    return _return_scalar_or_array(put_price, call)


def put_from_call(
    call_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    (
        call_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    ) = np.broadcast_arrays(
        _asarray(call_price),
        _asarray(stock_price),
        _asarray(strike_price),
        _asarray(risk_free_rate),
        _asarray(time_to_expiry),
        _asarray(dividend_yield),
    )

    _validate_positive(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")
    _validate_non_negative(call_price_arr, "call_price")

    discounted_stock = _discount_stock(stock_price_arr, dividend_yield_arr, time_to_expiry_arr)
    discounted_strike = _discount_strike(strike_price_arr, risk_free_rate_arr, time_to_expiry_arr)

    put = call_price_arr - discounted_stock + discounted_strike

    return _return_scalar_or_array(call_price, put)


def parity_residual(
    call_price: ArrayLike,
    put_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
):
    (
        call_price_arr,
        put_price_arr,
        stock_price_arr,
        strike_price_arr,
        risk_free_rate_arr,
        time_to_expiry_arr,
        dividend_yield_arr,
    ) = np.broadcast_arrays(
        _asarray(call_price),
        _asarray(put_price),
        _asarray(stock_price),
        _asarray(strike_price),
        _asarray(risk_free_rate),
        _asarray(time_to_expiry),
        _asarray(dividend_yield),
    )

    _validate_positive(stock_price_arr, "stock_price")
    _validate_positive(strike_price_arr, "strike_price")
    _validate_non_negative(time_to_expiry_arr, "time_to_expiry")

    discounted_stock = _discount_stock(stock_price_arr, dividend_yield_arr, time_to_expiry_arr)
    discounted_strike = _discount_strike(strike_price_arr, risk_free_rate_arr, time_to_expiry_arr)

    lhs = call_price_arr - put_price_arr
    rhs = discounted_stock - discounted_strike

    residual = lhs - rhs

    return _return_scalar_or_array(call_price, residual)


def verify_put_call_parity(
    call_price: ArrayLike,
    put_price: ArrayLike,
    stock_price: ArrayLike,
    strike_price: ArrayLike,
    risk_free_rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    tol: float = 1e-8,
):
    residual = parity_residual(
        call_price,
        put_price,
        stock_price,
        strike_price,
        risk_free_rate,
        time_to_expiry,
        dividend_yield,
    )

    residual_arr = _asarray(residual)
    satisfied = bool(np.all(np.abs(residual_arr) <= tol))

    return residual, satisfied
