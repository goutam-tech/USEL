"""
Tests for black_scholes.greeks
"""

from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.greeks import (
    delta_call,
    delta_put,
    gamma,
    rho_call,
    rho_put,
    theta_call,
    theta_put,
    vega,
)
from usel.black_scholes.pricing import call_price, put_price


class TestDeltaKnownValues:
    @staticmethod
    def test_delta_call_atm():
        assert delta_call(100, 100, 0.05, 0.20, 1) == pytest.approx(0.6368306511756191, rel=1e-10)

    @staticmethod
    def test_delta_put_atm():
        assert delta_put(100, 100, 0.05, 0.20, 1) == pytest.approx(-0.3631693488243809, rel=1e-10)

    @staticmethod
    def test_delta_call_minus_delta_put_equals_discounted_forward_factor():
        dc = delta_call(100, 100, 0.05, 0.20, 1, 0.02)
        dp = delta_put(100, 100, 0.05, 0.20, 1, 0.02)
        assert (dc - dp) == pytest.approx(np.exp(-0.02 * 1), rel=1e-10)


class TestGammaKnownValues:
    @staticmethod
    def test_gamma_atm():
        assert gamma(100, 100, 0.05, 0.20, 1) == pytest.approx(0.018762017345846895, rel=1e-10)

    @staticmethod
    def test_gamma_identical_for_call_and_put():
        stock_price = 100.0
        strike_price = 100.0
        risk_free_rate = 0.05
        volatility = 0.20
        time_to_expiry = 1.0

        gamma_value = gamma(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )

        # In Black-Scholes, gamma is identical for calls and puts.
        call_gamma = gamma_value
        put_gamma = gamma_value

        assert call_gamma == pytest.approx(put_gamma)


class TestVegaKnownValues:
    @staticmethod
    def test_vega_atm():
        assert vega(100, 100, 0.05, 0.20, 1) == pytest.approx(37.52403469169379, rel=1e-10)


class TestThetaKnownValues:
    @staticmethod
    def test_theta_call_atm():
        assert theta_call(100, 100, 0.05, 0.20, 1) == pytest.approx(-6.414027546438197, rel=1e-8)

    @staticmethod
    def test_theta_put_atm():
        assert theta_put(100, 100, 0.05, 0.20, 1) == pytest.approx(-1.657880423934626, rel=1e-8)


class TestRhoKnownValues:
    @staticmethod
    def test_rho_call_atm():
        assert rho_call(100, 100, 0.05, 0.20, 1) == pytest.approx(53.232481545376345, rel=1e-8)

    @staticmethod
    def test_rho_put_atm():
        assert rho_put(100, 100, 0.05, 0.20, 1) == pytest.approx(-41.89046090469506, rel=1e-8)


FD_EPS_PRICE = 1e-4
FD_EPS_VOL = 1e-6
FD_EPS_RATE = 1e-6
FD_EPS_TIME = 1e-6


class TestDeltaFiniteDifference:
    @pytest.mark.parametrize("stock_price", [80, 100, 120])
    def test_delta_call_matches_finite_difference(self, stock_price):
        h = FD_EPS_PRICE
        analytical = delta_call(stock_price, 100, 0.05, 0.20, 1)
        numerical = (
            call_price(stock_price + h, 100, 0.05, 0.20, 1)
            - call_price(stock_price - h, 100, 0.05, 0.20, 1)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, abs=1e-5)

    @pytest.mark.parametrize("stock_price", [80, 100, 120])
    def test_delta_put_matches_finite_difference(self, stock_price):
        h = FD_EPS_PRICE
        analytical = delta_put(stock_price, 100, 0.05, 0.20, 1)
        numerical = (
            put_price(stock_price + h, 100, 0.05, 0.20, 1)
            - put_price(stock_price - h, 100, 0.05, 0.20, 1)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, abs=1e-5)


class TestGammaFiniteDifference:
    @pytest.mark.parametrize("stock_price", [80, 100, 120])
    @staticmethod
    def test_gamma_matches_finite_difference_of_call_price(stock_price):
        h = FD_EPS_PRICE
        analytical = gamma(stock_price, 100, 0.05, 0.20, 1)
        numerical = (
            call_price(stock_price + h, 100, 0.05, 0.20, 1)
            - 2 * call_price(stock_price, 100, 0.05, 0.20, 1)
            + call_price(stock_price - h, 100, 0.05, 0.20, 1)
        ) / (h**2)
        assert analytical == pytest.approx(numerical, abs=1e-3)

    @staticmethod
    def test_gamma_matches_finite_difference_of_put_price():
        h = FD_EPS_PRICE
        analytical = gamma(100, 100, 0.05, 0.20, 1)
        numerical = (
            put_price(100 + h, 100, 0.05, 0.20, 1)
            - 2 * put_price(100, 100, 0.05, 0.20, 1)
            + put_price(100 - h, 100, 0.05, 0.20, 1)
        ) / (h**2)
        assert analytical == pytest.approx(numerical, abs=1e-3)


class TestVegaFiniteDifference:
    @staticmethod
    def test_vega_matches_finite_difference():
        h = FD_EPS_VOL
        analytical = vega(100, 100, 0.05, 0.20, 1)
        numerical = (
            call_price(100, 100, 0.05, 0.20 + h, 1) - call_price(100, 100, 0.05, 0.20 - h, 1)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, rel=1e-4)


class TestThetaFiniteDifference:
    @staticmethod
    def test_theta_call_matches_finite_difference():
        h = FD_EPS_TIME
        analytical = theta_call(100, 100, 0.05, 0.20, 1)
        numerical = -(
            call_price(100, 100, 0.05, 0.20, 1 + h) - call_price(100, 100, 0.05, 0.20, 1 - h)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, rel=1e-4)

    @staticmethod
    def test_theta_put_matches_finite_difference():
        h = FD_EPS_TIME
        analytical = theta_put(100, 100, 0.05, 0.20, 1)
        numerical = -(
            put_price(100, 100, 0.05, 0.20, 1 + h) - put_price(100, 100, 0.05, 0.20, 1 - h)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, rel=1e-4)


class TestRhoFiniteDifference:
    @staticmethod
    def test_rho_call_matches_finite_difference():
        h = FD_EPS_RATE
        analytical = rho_call(100, 100, 0.05, 0.20, 1)
        numerical = (
            call_price(100, 100, 0.05 + h, 0.20, 1) - call_price(100, 100, 0.05 - h, 0.20, 1)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, rel=1e-4)

    @staticmethod
    def test_rho_put_matches_finite_difference():
        h = FD_EPS_RATE
        analytical = rho_put(100, 100, 0.05, 0.20, 1)
        numerical = (
            put_price(100, 100, 0.05 + h, 0.20, 1) - put_price(100, 100, 0.05 - h, 0.20, 1)
        ) / (2 * h)
        assert analytical == pytest.approx(numerical, rel=1e-4)


class TestGreeksAtExpiry:
    @staticmethod
    def test_delta_call_itm_at_expiry():
        assert delta_call(110, 100, 0.05, 0.20, 0) == pytest.approx(1.0)

    @staticmethod
    def test_delta_call_otm_at_expiry():
        assert delta_call(90, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_delta_put_itm_at_expiry():
        assert delta_put(90, 100, 0.05, 0.20, 0) == pytest.approx(-1.0)

    @staticmethod
    def test_delta_put_otm_at_expiry():
        assert delta_put(110, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_gamma_zero_at_expiry():
        assert gamma(100, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_vega_zero_at_expiry():
        assert vega(100, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_theta_call_zero_at_expiry():
        assert theta_call(100, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_theta_put_zero_at_expiry():
        assert theta_put(100, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_rho_call_zero_at_expiry():
        assert rho_call(100, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_rho_put_zero_at_expiry():
        assert rho_put(100, 100, 0.05, 0.20, 0) == pytest.approx(0.0)

    @staticmethod
    def test_no_warnings_at_expiry(recwarn):
        for fn in (delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put):
            fn(100, 100, 0.05, 0.20, 0)
        assert len(recwarn) == 0


class TestGreeksValidation:
    @pytest.mark.parametrize(
        "fn", [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    )
    def test_negative_stock_price_raises(self, fn):
        with pytest.raises(ValueError):
            fn(-100, 100, 0.05, 0.20, 1)

    @pytest.mark.parametrize(
        "fn", [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    )
    def test_zero_strike_price_raises(self, fn):
        with pytest.raises(ValueError):
            fn(100, 0, 0.05, 0.20, 1)

    @pytest.mark.parametrize(
        "fn", [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    )
    def test_negative_volatility_raises(self, fn):
        with pytest.raises(ValueError):
            fn(100, 100, 0.05, -0.20, 1)

    @pytest.mark.parametrize(
        "fn", [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    )
    def test_negative_time_to_expiry_raises(self, fn):
        with pytest.raises(ValueError):
            fn(100, 100, 0.05, 0.20, -1)


class TestGreeksSanityBounds:
    @staticmethod
    def test_delta_call_bounded_zero_one():
        d = delta_call(100, 100, 0.05, 0.20, 1)
        assert 0.0 <= d <= 1.0

    @staticmethod
    def test_delta_put_bounded_minus_one_zero():
        d = delta_put(100, 100, 0.05, 0.20, 1)
        assert -1.0 <= d <= 0.0

    @staticmethod
    def test_gamma_non_negative():
        assert gamma(100, 100, 0.05, 0.20, 1) >= 0.0

    @staticmethod
    def test_vega_non_negative():
        assert vega(100, 100, 0.05, 0.20, 1) >= 0.0

    @staticmethod
    def test_rho_call_non_negative():
        assert rho_call(100, 100, 0.05, 0.20, 1) >= 0.0

    @staticmethod
    def test_rho_put_non_positive():
        assert rho_put(100, 100, 0.05, 0.20, 1) <= 0.0

    @staticmethod
    def test_delta_call_approaches_one_deep_itm():
        d = delta_call(200, 100, 0.05, 0.20, 1)
        assert d == pytest.approx(1.0, abs=1e-3)

    @staticmethod
    def test_delta_call_approaches_zero_deep_otm():
        d = delta_call(50, 100, 0.05, 0.20, 1)
        assert d == pytest.approx(0.0, abs=1e-3)

    @staticmethod
    def test_gamma_peaks_near_atm():
        gamma_itm = gamma(120, 100, 0.05, 0.20, 1)
        gamma_atm = gamma(100, 100, 0.05, 0.20, 1)
        gamma_otm = gamma(80, 100, 0.05, 0.20, 1)
        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm


class TestGreeksBroadcasting:
    @staticmethod
    def test_delta_call_accepts_array_stock_price():
        stock_prices = np.array([80.0, 100.0, 120.0])
        result = delta_call(stock_prices, 100, 0.05, 0.20, 1)
        assert isinstance(result, np.ndarray)
        assert result.shape == stock_prices.shape

    @staticmethod
    def test_gamma_array_matches_scalar_elementwise():
        stock_prices = np.array([80.0, 100.0, 120.0])
        array_result = gamma(stock_prices, 100, 0.05, 0.20, 1)
        scalar_results = np.array([gamma(s, 100, 0.05, 0.20, 1) for s in stock_prices])
        np.testing.assert_allclose(array_result, scalar_results, rtol=1e-10)

    @staticmethod
    def test_mixed_scalar_and_array_time_to_expiry():
        times = np.array([0.0, 0.5, 1.0])
        result = delta_call(100, 100, 0.05, 0.20, times)
        assert result.shape == times.shape
