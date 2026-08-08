from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.parity import (
    call_from_put,
    parity_residual,
    put_from_call,
    verify_put_call_parity,
)
from usel.black_scholes.pricing import call_price, put_price


class TestCallFromPut:
    @staticmethod
    def test_recovers_known_call_value():
        recovered_call = call_from_put(5.573526022256971, 100, 100, 0.05, 1)
        assert recovered_call == pytest.approx(10.450583572185565, rel=1e-10)

    @staticmethod
    def test_round_trip_with_pricing_module():
        call = call_price(100, 100, 0.05, 0.20, 1)
        put = put_price(100, 100, 0.05, 0.20, 1)

        recovered_call = call_from_put(put, 100, 100, 0.05, 1)
        assert recovered_call == pytest.approx(call, rel=1e-10)

    @staticmethod
    def test_round_trip_with_dividend_yield():
        call = call_price(100, 100, 0.05, 0.20, 1, 0.02)
        put = put_price(100, 100, 0.05, 0.20, 1, 0.02)

        recovered_call = call_from_put(put, 100, 100, 0.05, 1, 0.02)
        assert recovered_call == pytest.approx(call, rel=1e-10)

    @staticmethod
    def test_returns_float_for_scalar_input():
        result = call_from_put(5.573526022256971, 100, 100, 0.05, 1)
        assert isinstance(result, float)


class TestPutFromCall:
    @staticmethod
    def test_recovers_known_put_value():
        recovered_put = put_from_call(10.450583572185565, 100, 100, 0.05, 1)
        assert recovered_put == pytest.approx(5.573526022256971, rel=1e-10)

    @staticmethod
    def test_round_trip_with_pricing_module():
        call = call_price(100, 100, 0.05, 0.20, 1)
        put = put_price(100, 100, 0.05, 0.20, 1)

        recovered_put = put_from_call(call, 100, 100, 0.05, 1)
        assert recovered_put == pytest.approx(put, rel=1e-10)

    @staticmethod
    def test_round_trip_with_dividend_yield():
        call = call_price(100, 100, 0.05, 0.20, 1, 0.02)
        put = put_price(100, 100, 0.05, 0.20, 1, 0.02)

        recovered_put = put_from_call(call, 100, 100, 0.05, 1, 0.02)
        assert recovered_put == pytest.approx(put, rel=1e-10)


class TestRoundTripConsistency:
    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry,dividend_yield",
        [
            (100, 100, 0.05, 0.20, 1, 0.0),
            (80, 100, 0.03, 0.35, 0.5, 0.0),
            (120, 90, 0.01, 0.15, 2.0, 0.02),
            (50, 60, 0.10, 0.40, 0.25, 0.01),
        ],
    )
    def test_call_and_put_recover_each_other(
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

        assert call_from_put(
            put,
            stock_price,
            strike_price,
            risk_free_rate,
            time_to_expiry,
            dividend_yield,
        ) == pytest.approx(call, rel=1e-10)

        assert put_from_call(
            call,
            stock_price,
            strike_price,
            risk_free_rate,
            time_to_expiry,
            dividend_yield,
        ) == pytest.approx(put, rel=1e-10)


class TestParityResidual:
    @staticmethod
    def test_residual_near_zero_for_consistent_prices():
        call = call_price(100, 100, 0.05, 0.20, 1)
        put = put_price(100, 100, 0.05, 0.20, 1)

        residual = parity_residual(call, put, 100, 100, 0.05, 1)
        assert residual == pytest.approx(0.0, abs=1e-8)

    @staticmethod
    def test_residual_nonzero_for_inconsistent_prices():
        residual = parity_residual(10.0, 3.0, 100, 100, 0.05, 1)
        assert abs(residual) > 1e-3

    @staticmethod
    def test_residual_scales_with_price_perturbation():
        call = call_price(100, 100, 0.05, 0.20, 1)
        put = put_price(100, 100, 0.05, 0.20, 1)

        residual = parity_residual(call + 1.0, put, 100, 100, 0.05, 1)
        assert residual == pytest.approx(1.0, rel=1e-8)


class TestVerifyPutCallParity:
    @staticmethod
    def test_satisfied_true_for_consistent_prices():
        call = call_price(100, 100, 0.05, 0.20, 1)
        put = put_price(100, 100, 0.05, 0.20, 1)

        residual, satisfied = verify_put_call_parity(call, put, 100, 100, 0.05, 1)
        assert satisfied is True
        assert residual == pytest.approx(0.0, abs=1e-8)

    @staticmethod
    def test_satisfied_false_for_inconsistent_prices():
        _, satisfied = verify_put_call_parity(10.0, 3.0, 100, 100, 0.05, 1)
        assert satisfied is False

    @staticmethod
    def test_returns_tuple():
        result = verify_put_call_parity(10.0, 3.0, 100, 100, 0.05, 1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @staticmethod
    def test_custom_tolerance_can_flip_result():
        call = call_price(100, 100, 0.05, 0.20, 1)
        put = put_price(100, 100, 0.05, 0.20, 1)

        _, satisfied_tight = verify_put_call_parity(call + 1e-6, put, 100, 100, 0.05, 1, tol=1e-9)
        _, satisfied_loose = verify_put_call_parity(call + 1e-6, put, 100, 100, 0.05, 1, tol=1e-5)
        assert satisfied_tight is False
        assert satisfied_loose is True


class TestParityValidation:
    @staticmethod
    def test_call_from_put_rejects_negative_stock_price():
        with pytest.raises(ValueError):
            call_from_put(5.0, -100, 100, 0.05, 1)

    @staticmethod
    def test_call_from_put_rejects_zero_strike_price():
        with pytest.raises(ValueError):
            call_from_put(5.0, 100, 0, 0.05, 1)

    @staticmethod
    def test_call_from_put_rejects_negative_time_to_expiry():
        with pytest.raises(ValueError):
            call_from_put(5.0, 100, 100, 0.05, -1)

    @staticmethod
    def test_put_from_call_rejects_negative_strike_price():
        with pytest.raises(ValueError):
            put_from_call(10.0, 100, -100, 0.05, 1)

    @staticmethod
    def test_parity_residual_rejects_negative_stock_price():
        with pytest.raises(ValueError):
            parity_residual(10.0, 5.0, -100, 100, 0.05, 1)


class TestParityBroadcasting:
    @staticmethod
    def test_call_from_put_accepts_arrays():
        put_prices = np.array([3.0, 5.0, 8.0])
        result = call_from_put(put_prices, 100, 100, 0.05, 1)

        assert isinstance(result, np.ndarray)
        assert result.shape == put_prices.shape

    @staticmethod
    def test_verify_put_call_parity_with_arrays():
        stock_prices = np.array([90.0, 100.0, 110.0])
        calls = call_price(stock_prices, 100, 0.05, 0.20, 1)
        puts = put_price(stock_prices, 100, 0.05, 0.20, 1)

        residual, satisfied = verify_put_call_parity(calls, puts, stock_prices, 100, 0.05, 1)

        assert residual.shape == stock_prices.shape
        assert satisfied is True
