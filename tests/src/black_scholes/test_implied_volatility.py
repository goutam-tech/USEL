"""
Tests for black_scholes.implied_volatility
"""

from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.implied_volatility import (
    implied_volatility,
    implied_volatility_call,
    implied_volatility_put,
)
from usel.black_scholes.pricing import call_price, put_price


class TestImpliedVolatilityCallRoundTrip:
    @pytest.mark.parametrize("true_vol", [0.05, 0.10, 0.20, 0.35, 0.50, 1.0])
    @staticmethod
    def test_recovers_true_volatility(true_vol):
        price = call_price(100, 100, 0.05, true_vol, 1)
        recovered_vol = implied_volatility_call(price, 100, 100, 0.05, 1)
        assert recovered_vol == pytest.approx(true_vol, abs=1e-6)

    @pytest.mark.parametrize(
        "stock_price,strike_price", [(80, 100), (100, 100), (120, 100), (100, 80)]
    )
    @staticmethod
    def test_recovers_true_volatility_across_moneyness(stock_price, strike_price):
        true_vol = 0.25
        price = call_price(stock_price, strike_price, 0.05, true_vol, 1)
        recovered_vol = implied_volatility_call(price, stock_price, strike_price, 0.05, 1)
        assert recovered_vol == pytest.approx(true_vol, abs=1e-6)

    @staticmethod
    def test_recovers_true_volatility_with_dividend_yield():
        true_vol = 0.30
        price = call_price(100, 100, 0.05, true_vol, 1, 0.03)
        recovered_vol = implied_volatility_call(price, 100, 100, 0.05, 1, 0.03)
        assert recovered_vol == pytest.approx(true_vol, abs=1e-6)

    @staticmethod
    def test_recovers_true_volatility_short_dated():
        true_vol = 0.40
        price = call_price(100, 100, 0.05, true_vol, 0.05)
        recovered_vol = implied_volatility_call(price, 100, 100, 0.05, 0.05)
        assert recovered_vol == pytest.approx(true_vol, abs=1e-5)


class TestImpliedVolatilityPutRoundTrip:
    @pytest.mark.parametrize("true_vol", [0.05, 0.10, 0.20, 0.35, 0.50, 1.0])
    def test_recovers_true_volatility(self, true_vol):
        price = put_price(100, 100, 0.05, true_vol, 1)
        recovered_vol = implied_volatility_put(price, 100, 100, 0.05, 1)
        assert recovered_vol == pytest.approx(true_vol, abs=1e-6)

    @staticmethod
    def test_recovers_true_volatility_with_dividend_yield():
        true_vol = 0.30
        price = put_price(100, 100, 0.05, true_vol, 1, 0.03)
        recovered_vol = implied_volatility_put(price, 100, 100, 0.05, 1, 0.03)
        assert recovered_vol == pytest.approx(true_vol, abs=1e-6)


class TestImpliedVolatilityDispatcher:
    @staticmethod
    def test_dispatches_to_call():
        price = call_price(100, 100, 0.05, 0.20, 1)
        assert implied_volatility(price, 100, 100, 0.05, 1, "call") == pytest.approx(0.20, abs=1e-6)

    @staticmethod
    def test_dispatches_to_put():
        price = put_price(100, 100, 0.05, 0.20, 1)
        assert implied_volatility(price, 100, 100, 0.05, 1, "put") == pytest.approx(0.20, abs=1e-6)

    @staticmethod
    def test_case_insensitive_option_type():
        price = call_price(100, 100, 0.05, 0.20, 1)
        assert implied_volatility(price, 100, 100, 0.05, 1, "CALL") == pytest.approx(0.20, abs=1e-6)

    @staticmethod
    def test_invalid_option_type_raises():
        with pytest.raises(ValueError):
            implied_volatility(10.0, 100, 100, 0.05, 1, "straddle")


class TestNoArbitrageBounds:
    @staticmethod
    def test_call_price_below_intrinsic_raises():
        with pytest.raises(ValueError):
            implied_volatility_call(1.0, 120, 100, 0.05, 1)

    @staticmethod
    def test_call_price_above_stock_price_raises():
        with pytest.raises(ValueError):
            implied_volatility_call(150.0, 100, 100, 0.05, 1)

    @staticmethod
    def test_put_price_below_intrinsic_raises():
        with pytest.raises(ValueError):
            implied_volatility_put(1.0, 80, 100, 0.05, 1)

    @staticmethod
    def test_put_price_above_discounted_strike_raises():
        with pytest.raises(ValueError):
            implied_volatility_put(150.0, 100, 100, 0.05, 1)

    @staticmethod
    def test_negative_market_price_raises():
        with pytest.raises(ValueError):
            implied_volatility_call(-1.0, 100, 100, 0.05, 1)

    @staticmethod
    def test_zero_time_to_expiry_raises():
        with pytest.raises(ValueError):
            implied_volatility_call(10.0, 100, 100, 0.05, 0)

    @staticmethod
    def test_negative_time_to_expiry_raises():
        with pytest.raises(ValueError):
            implied_volatility_call(10.0, 100, 100, 0.05, -1)

    @staticmethod
    def test_boundary_price_just_inside_bounds_succeeds():
        intrinsic = max(100 * np.exp(0) - 100 * np.exp(-0.05 * 1), 0.0)
        price = intrinsic + 1e-6
        vol = implied_volatility_call(price, 100, 100, 0.05, 1)
        assert vol >= 0.0


class TestConvergenceBehavior:
    @staticmethod
    def test_converges_regardless_of_initial_guess():
        true_vol = 0.20
        price = call_price(100, 100, 0.05, true_vol, 1)

        for guess in [0.01, 0.1, 0.5, 1.0, 4.0]:
            vol = implied_volatility_call(price, 100, 100, 0.05, 1, initial_guess=guess)
            assert vol == pytest.approx(true_vol, abs=1e-5)

    @staticmethod
    def test_converges_for_deep_otm_low_vega():
        true_vol = 0.15
        price = call_price(60, 100, 0.05, true_vol, 0.25)
        vol = implied_volatility_call(price, 60, 100, 0.05, 0.25)
        assert vol == pytest.approx(true_vol, abs=1e-4)

    @staticmethod
    def test_converges_for_deep_itm_low_vega():
        true_vol = 0.15
        price = call_price(140, 100, 0.05, true_vol, 0.25)
        vol = implied_volatility_call(price, 140, 100, 0.05, 0.25)
        assert vol == pytest.approx(true_vol, abs=1e-4)

    @staticmethod
    def test_tighter_tolerance_improves_accuracy():
        true_vol = 0.20
        price = call_price(100, 100, 0.05, true_vol, 1)

        loose = implied_volatility_call(price, 100, 100, 0.05, 1, tol=1e-3)
        tight = implied_volatility_call(price, 100, 100, 0.05, 1, tol=1e-10)

        assert abs(tight - true_vol) <= abs(loose - true_vol) + 1e-12

    @staticmethod
    def test_warns_when_max_iterations_too_low():
        price = call_price(100, 100, 0.05, 0.20, 1)
        with pytest.warns(RuntimeWarning):
            implied_volatility_call(price, 100, 100, 0.05, 1, max_iterations=1)


class TestImpliedVolatilityBroadcasting:
    @staticmethod
    def test_accepts_array_of_market_prices():
        true_vols = np.array([0.10, 0.20, 0.30, 0.40])
        prices = call_price(100, 100, 0.05, true_vols, 1)

        recovered = implied_volatility_call(prices, 100, 100, 0.05, 1)

        assert isinstance(recovered, np.ndarray)
        np.testing.assert_allclose(recovered, true_vols, atol=1e-6)

    @staticmethod
    def test_accepts_array_of_strikes():
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        true_vol = 0.25
        prices = call_price(100, strikes, 0.05, true_vol, 1)

        recovered = implied_volatility_call(prices, 100, strikes, 0.05, 1)

        assert recovered.shape == strikes.shape
        np.testing.assert_allclose(recovered, np.full_like(strikes, true_vol), atol=1e-6)

    @staticmethod
    def test_mixed_scalar_and_array_inputs():
        stock_prices = np.array([90.0, 100.0, 110.0])
        true_vol = 0.20
        prices = call_price(stock_prices, 100, 0.05, true_vol, 1)

        recovered = implied_volatility_call(prices, stock_prices, 100, 0.05, 1)

        assert recovered.shape == stock_prices.shape
        np.testing.assert_allclose(recovered, np.full(3, true_vol), atol=1e-6)
