from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.pricing import call_price, put_price


class TestCallPriceKnownValues:
    @staticmethod
    def test_atm_call_no_dividend():
        price = call_price(100, 100, 0.05, 0.20, 1)
        assert price == pytest.approx(10.450583572185565, rel=1e-10)

    @staticmethod
    def test_call_price_is_float():
        price = call_price(100, 100, 0.05, 0.20, 1)
        assert isinstance(price, float)


class TestPutPriceKnownValues:
    @staticmethod
    def test_atm_put_no_dividend():
        price = put_price(100, 100, 0.05, 0.20, 1)
        assert price == pytest.approx(5.573526022256971, rel=1e-10)

    @staticmethod
    def test_put_price_is_float():
        price = put_price(100, 100, 0.05, 0.20, 1)
        assert isinstance(price, float)


class TestPutCallParity:
    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry,dividend_yield",
        [
            (100, 100, 0.05, 0.20, 1, 0.0),
            (100, 90, 0.03, 0.35, 0.5, 0.0),
            (50, 60, 0.01, 0.40, 2.0, 0.0),
            (120, 100, 0.05, 0.20, 1.0, 0.02),
            (80, 100, 0.10, 0.15, 0.25, 0.03),
        ],
    )
    def test_parity_holds(
        self,
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    ):
        call = call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        put = put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )

        lhs = call - put
        rhs = stock_price * np.exp(-dividend_yield * time_to_expiry) - strike_price * np.exp(
            -risk_free_rate * time_to_expiry
        )

        assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


class TestZeroTimeToExpiry:
    @staticmethod
    def test_call_itm_at_expiry():
        price = call_price(110, 100, 0.05, 0.20, 0)
        assert price == pytest.approx(10.0, rel=1e-10)

    @staticmethod
    def test_call_otm_at_expiry():
        price = call_price(90, 100, 0.05, 0.20, 0)
        assert price == pytest.approx(0.0, abs=1e-10)

    @staticmethod
    def test_put_itm_at_expiry():
        price = put_price(90, 100, 0.05, 0.20, 0)
        assert price == pytest.approx(10.0, rel=1e-10)

    @staticmethod
    def test_put_otm_at_expiry():
        price = put_price(110, 100, 0.05, 0.20, 0)
        assert price == pytest.approx(0.0, abs=1e-10)

    @staticmethod
    def test_call_atm_at_expiry():
        price = call_price(100, 100, 0.05, 0.20, 0)
        assert price == pytest.approx(0.0, abs=1e-10)

    @staticmethod
    def test_no_nan_or_inf_at_expiry():
        call = call_price(100, 100, 0.05, 0.20, 0)
        put = put_price(100, 100, 0.05, 0.20, 0)
        assert np.isfinite(call)
        assert np.isfinite(put)

    @staticmethod
    def test_no_warnings_at_expiry(recwarn):
        call_price(100, 100, 0.05, 0.20, 0)
        put_price(100, 100, 0.05, 0.20, 0)
        assert len(recwarn) == 0


class TestCallPriceValidation:
    @staticmethod
    def test_negative_stock_price_raises():
        with pytest.raises(ValueError):
            call_price(-100, 100, 0.05, 0.20, 1)

    @staticmethod
    def test_zero_stock_price_raises():
        with pytest.raises(ValueError):
            call_price(0, 100, 0.05, 0.20, 1)

    @staticmethod
    def test_negative_strike_price_raises():
        with pytest.raises(ValueError):
            call_price(100, -100, 0.05, 0.20, 1)

    @staticmethod
    def test_zero_strike_price_raises():
        with pytest.raises(ValueError):
            call_price(100, 0, 0.05, 0.20, 1)

    @staticmethod
    def test_negative_volatility_raises():
        with pytest.raises(ValueError):
            call_price(100, 100, 0.05, -0.20, 1)

    @staticmethod
    def test_zero_volatility_raises():
        with pytest.raises(ValueError):
            call_price(100, 100, 0.05, 0.0, 1)

    @staticmethod
    def test_negative_time_to_expiry_raises():
        with pytest.raises(ValueError):
            call_price(100, 100, 0.05, 0.20, -1)


class TestPutPriceValidation:
    @staticmethod
    def test_negative_stock_price_raises():
        with pytest.raises(ValueError):
            put_price(-100, 100, 0.05, 0.20, 1)

    @staticmethod
    def test_zero_strike_price_raises():
        with pytest.raises(ValueError):
            put_price(100, 0, 0.05, 0.20, 1)

    @staticmethod
    def test_negative_volatility_raises():
        with pytest.raises(ValueError):
            put_price(100, 100, 0.05, -0.20, 1)

    @staticmethod
    def test_negative_time_to_expiry_raises():
        with pytest.raises(ValueError):
            put_price(100, 100, 0.05, 0.20, -1)


class TestDividendYield:
    @staticmethod
    def test_default_dividend_yield_is_zero():
        with_default = call_price(100, 100, 0.05, 0.20, 1)
        explicit_zero = call_price(100, 100, 0.05, 0.20, 1, 0.0)
        assert with_default == pytest.approx(explicit_zero, rel=1e-12)

    @staticmethod
    def test_higher_dividend_yield_lowers_call_price():
        low_q = call_price(100, 100, 0.05, 0.20, 1, 0.0)
        high_q = call_price(100, 100, 0.05, 0.20, 1, 0.05)
        assert high_q < low_q

    @staticmethod
    def test_higher_dividend_yield_raises_put_price():
        low_q = put_price(100, 100, 0.05, 0.20, 1, 0.0)
        high_q = put_price(100, 100, 0.05, 0.20, 1, 0.05)
        assert high_q > low_q


class TestBroadcasting:
    @staticmethod
    def test_call_price_accepts_array_stock_price():
        stock_prices = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = call_price(stock_prices, 100, 0.05, 0.20, 1)

        assert isinstance(prices, np.ndarray)
        assert prices.shape == stock_prices.shape

    def test_call_price_array_matches_scalar_elementwise(self):
        stock_prices = np.array([80.0, 100.0, 120.0])
        array_result = call_price(stock_prices, 100, 0.05, 0.20, 1)

        scalar_results = np.array([call_price(s, 100, 0.05, 0.20, 1) for s in stock_prices])

        np.testing.assert_allclose(array_result, scalar_results, rtol=1e-10)

    @staticmethod
    def test_put_price_accepts_array_strike_price():
        stock_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        strike_prices = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

        prices = put_price(
            stock_prices,
            strike_prices,
            0.05,
            0.20,
            1,
        )

        assert isinstance(prices, np.ndarray)
        assert prices.shape == strike_prices.shape

    @staticmethod
    def test_fully_broadcast_multi_parameter_arrays():
        stock_prices = np.array([90.0, 100.0, 110.0])
        volatilities = np.array([0.15, 0.20, 0.25])

        prices = call_price(stock_prices, 100, 0.05, volatilities, 1)

        assert prices.shape == (3,)
        assert np.all(np.isfinite(prices))

    @staticmethod
    def test_mixed_scalar_and_array_time_to_expiry():
        stock_prices = np.array([100.0, 100.0, 100.0])
        times = np.array([0.0, 0.5, 1.0])

        prices = call_price(
            stock_prices,
            100,
            0.05,
            0.20,
            times,
        )

        assert prices.shape == times.shape
        assert prices[0] == pytest.approx(0.0, abs=1e-10)


class TestSanityBounds:
    @staticmethod
    def test_call_price_non_negative():
        price = call_price(100, 100, 0.05, 0.20, 1)
        assert price >= 0.0

    @staticmethod
    def test_put_price_non_negative():
        price = put_price(100, 100, 0.05, 0.20, 1)
        assert price >= 0.0

    @staticmethod
    def test_deep_itm_call_approaches_intrinsic():
        price = call_price(150, 100, 0.05, 0.01, 0.001)
        assert price == pytest.approx(50.0, rel=1e-2)

    @staticmethod
    def test_deep_otm_call_near_zero():
        price = call_price(50, 100, 0.05, 0.10, 0.1)
        assert price == pytest.approx(0.0, abs=1e-3)

    @staticmethod
    def test_deep_itm_put_approaches_intrinsic():
        price = put_price(50, 100, 0.05, 0.01, 0.001)
        assert price == pytest.approx(50.0, rel=1e-2)

    @staticmethod
    def test_call_price_bounded_above_by_stock_price():
        price = call_price(100, 100, 0.05, 0.20, 1)
        assert price <= 100.0

    @staticmethod
    def test_put_price_bounded_above_by_discounted_strike():
        price = put_price(100, 100, 0.05, 0.20, 1)
        assert price <= 100 * np.exp(-0.05 * 1)


class TestMonotonicity:
    @staticmethod
    def test_call_price_increases_with_stock_price():
        prices = [call_price(s, 100, 0.05, 0.20, 1) for s in [80, 90, 100, 110, 120]]
        assert prices == sorted(prices)

    @staticmethod
    def test_put_price_decreases_with_stock_price():
        prices = [put_price(s, 100, 0.05, 0.20, 1) for s in [80, 90, 100, 110, 120]]
        assert prices == sorted(prices, reverse=True)

    @staticmethod
    def test_call_price_increases_with_volatility():
        prices = [call_price(100, 100, 0.05, sigma, 1) for sigma in [0.1, 0.2, 0.3, 0.4]]
        assert prices == sorted(prices)

    @staticmethod
    def test_put_price_increases_with_volatility():
        prices = [put_price(100, 100, 0.05, sigma, 1) for sigma in [0.1, 0.2, 0.3, 0.4]]
        assert prices == sorted(prices)

    @staticmethod
    def test_call_price_increases_with_time_to_expiry():
        prices = [call_price(100, 100, 0.05, 0.20, t) for t in [0.25, 0.5, 1.0, 2.0]]
        assert prices == sorted(prices)
