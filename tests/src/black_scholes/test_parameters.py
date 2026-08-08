from __future__ import annotations

import math

import numpy as np
import pytest

from usel.black_scholes.parameters import d1, d2


@pytest.mark.parametrize(
    ("stock", "strike", "rate", "volatility", "time", "expected"),
    [
        (100.0, 100.0, 0.05, 0.20, 1.0, 0.35),
        (50.0, 45.0, 0.03, 0.25, 2.0, 0.644487),
    ],
)
def test_d1_known_values(
    stock: float,
    strike: float,
    rate: float,
    volatility: float,
    time: float,
    expected: float,
) -> None:
    result = d1(
        stock,
        strike,
        rate,
        volatility,
        time,
    )

    assert result == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    ("stock", "strike", "rate", "volatility", "time", "expected"),
    [
        (100.0, 100.0, 0.05, 0.20, 1.0, 0.15),
        (50.0, 45.0, 0.03, 0.25, 2.0, 0.290933),
    ],
)
def test_d2_known_values(
    stock: float,
    strike: float,
    rate: float,
    volatility: float,
    time: float,
    expected: float,
) -> None:
    result = d2(
        stock,
        strike,
        rate,
        volatility,
        time,
    )

    assert result == pytest.approx(expected, rel=1e-5)


def test_d1_array() -> None:
    stock = np.array([100.0, 110.0])
    strike = np.array([100.0, 100.0])

    result = d1(
        stock,
        strike,
        0.05,
        0.20,
        1.0,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_d2_array() -> None:
    stock = np.array([100.0, 110.0])
    strike = np.array([100.0, 100.0])

    result = d2(
        stock,
        strike,
        0.05,
        0.20,
        1.0,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_broadcasting_inputs() -> None:
    stock = np.array([90.0, 100.0, 110.0])

    result = d1(
        stock,
        100.0,
        0.05,
        0.20,
        1.0,
    )

    assert result.shape == stock.shape


@pytest.mark.parametrize(
    "stock",
    [80.0, 100.0, 120.0],
)
def test_d2_identity(stock: float) -> None:
    volatility = 0.30
    time = 2.0

    d1_value = d1(
        stock,
        100.0,
        0.05,
        volatility,
        time,
    )

    d2_value = d2(
        stock,
        100.0,
        0.05,
        volatility,
        time,
    )

    expected = d1_value - volatility * math.sqrt(time)

    assert d2_value == pytest.approx(expected)


def test_dividend_yield_changes_result() -> None:
    without_dividend = d1(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
    )

    with_dividend = d1(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
        dividend_yield=0.03,
    )

    assert with_dividend < without_dividend


def test_zero_expiry_above_strike() -> None:
    value = d1(
        120.0,
        100.0,
        0.05,
        0.20,
        0.0,
    )

    assert value == np.inf


def test_zero_expiry_below_strike() -> None:
    value = d1(
        80.0,
        100.0,
        0.05,
        0.20,
        0.0,
    )

    assert value == -np.inf


def test_zero_expiry_at_money() -> None:
    value = d1(
        100.0,
        100.0,
        0.05,
        0.20,
        0.0,
    )

    assert value == 0.0


def test_d2_zero_expiry() -> None:
    value = d2(
        100.0,
        100.0,
        0.05,
        0.20,
        0.0,
    )

    assert value == 0.0


@pytest.mark.parametrize(
    "stock",
    [0.0, -1.0],
)
def test_invalid_stock(stock: float) -> None:
    with pytest.raises(ValueError):
        d1(
            stock,
            100.0,
            0.05,
            0.20,
            1.0,
        )


@pytest.mark.parametrize(
    "strike",
    [0.0, -100.0],
)
def test_invalid_strike(strike: float) -> None:
    with pytest.raises(ValueError):
        d1(
            100.0,
            strike,
            0.05,
            0.20,
            1.0,
        )


@pytest.mark.parametrize(
    "volatility",
    [0.0, -0.2],
)
def test_invalid_volatility(volatility: float) -> None:
    with pytest.raises(ValueError):
        d1(
            100.0,
            100.0,
            0.05,
            volatility,
            1.0,
        )


def test_negative_time() -> None:
    with pytest.raises(ValueError):
        d1(
            100.0,
            100.0,
            0.05,
            0.20,
            -1.0,
        )


def test_shape_preservation() -> None:
    stock = np.random.rand(10, 5) * 100 + 50

    result = d1(
        stock,
        100.0,
        0.05,
        0.20,
        1.0,
    )

    assert result.shape == stock.shape


def test_scalar_return_type() -> None:
    value = d1(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
    )

    assert isinstance(value, float)


def test_d1_greater_than_d2() -> None:
    d1_value = d1(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
    )

    d2_value = d2(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
    )

    assert d1_value > d2_value


def test_d1_equals_d2_when_time_zero() -> None:
    value1 = d1(
        120.0,
        100.0,
        0.05,
        0.20,
        0.0,
    )

    value2 = d2(
        120.0,
        100.0,
        0.05,
        0.20,
        0.0,
    )

    assert value1 == value2
