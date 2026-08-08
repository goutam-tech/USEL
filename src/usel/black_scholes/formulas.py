import numpy as np


def d1(
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
):
    return (
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))


def d2(
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
):
    return d1(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    ) - volatility * np.sqrt(time_to_expiry)
