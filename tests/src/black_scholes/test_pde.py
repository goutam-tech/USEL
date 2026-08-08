"""
Tests for black_scholes.pde
"""

from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.pde import (
    black_scholes_pde_residual,
    call_boundary_condition_lower,
    call_boundary_condition_upper,
    put_boundary_condition_lower,
    put_boundary_condition_upper,
    terminal_condition_call,
    terminal_condition_put,
    verify_black_scholes_pde,
)
from usel.black_scholes.pricing import call_price, put_price


class TestPdeResidual:
    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry,dividend_yield",
        [
            (100, 100, 0.05, 0.20, 1.0, 0.0),
            (80, 100, 0.03, 0.35, 0.5, 0.0),
            (120, 90, 0.01, 0.15, 2.0, 0.02),
            (50, 60, 0.10, 0.40, 0.25, 0.01),
            (150, 100, 0.07, 0.50, 0.05, 0.0),
        ],
    )
    def test_call_residual_near_zero(
        self,
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    ):
        residual = black_scholes_pde_residual(
            "call",
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        assert residual == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry,dividend_yield",
        [
            (100, 100, 0.05, 0.20, 1.0, 0.0),
            (80, 100, 0.03, 0.35, 0.5, 0.0),
        ]
    )
    @pytest.mark.parametrize(
        "stock_price, strike_price, risk_free_rate, volatility, time_to_expiry, dividend_yield",
        [
            (100, 100, 0.05, 0.20, 1, 0.0),
            (120, 90, 0.01, 0.15, 2.0, 0.02),
            (50, 60, 0.10, 0.40, 0.25, 0.01),
            (150, 100, 0.07, 0.50, 0.05, 0.0),
        ],
    )
    @staticmethod
    def test_put_residual_near_zero(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    ):
        residual = black_scholes_pde_residual(
            "put",
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            dividend_yield,
        )
        assert residual == pytest.approx(0.0, abs=1e-8)

    @staticmethod
    def test_case_insensitive_option_type():
        residual_lower = black_scholes_pde_residual("call", 100, 100, 0.05, 0.20, 1)
        residual_upper = black_scholes_pde_residual("CALL", 100, 100, 0.05, 0.20, 1)
        assert residual_lower == pytest.approx(residual_upper, abs=1e-12)

    @staticmethod
    def test_invalid_option_type_raises():
        with pytest.raises(ValueError):
            black_scholes_pde_residual("straddle", 100, 100, 0.05, 0.20, 1)


class TestVerifyBlackScholesPde:
    @staticmethod
    def test_call_satisfied_true():
        residual, satisfied = verify_black_scholes_pde("call", 100, 100, 0.05, 0.20, 1)
        assert satisfied is True
        assert residual == pytest.approx(0.0, abs=1e-6)

    @staticmethod
    def test_put_satisfied_true():
        residual, satisfied = verify_black_scholes_pde("put", 100, 100, 0.05, 0.20, 1)
        assert satisfied is True
        assert residual == pytest.approx(0.0, abs=1e-6)

    @staticmethod
    def test_returns_tuple():
        result = verify_black_scholes_pde("call", 100, 100, 0.05, 0.20, 1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @staticmethod
    def test_custom_tight_tolerance_still_satisfied():
        _, satisfied = verify_black_scholes_pde("call", 100, 100, 0.05, 0.20, 1, tol=1e-6)
        assert satisfied is True

    @staticmethod
    def test_satisfied_with_dividend_yield():
        _, satisfied = verify_black_scholes_pde(
            "call", 100, 100, 0.05, 0.20, 1, dividend_yield=0.03
        )
        assert satisfied is True

    @staticmethod
    def test_satisfied_at_various_moneyness():
        for stock_price in [50, 80, 100, 120, 200]:
            _, satisfied = verify_black_scholes_pde("call", stock_price, 100, 0.05, 0.20, 1)
            assert satisfied is True

    @staticmethod
    def test_array_input_all_satisfied():
        stock_prices = np.array([80.0, 100.0, 120.0])
        residual, satisfied = verify_black_scholes_pde("call", stock_prices, 100, 0.05, 0.20, 1)
        assert residual.shape == stock_prices.shape
        assert satisfied is True


class TestTerminalConditionCall:
    @staticmethod
    def test_itm_payoff():
        assert terminal_condition_call(110, 100) == pytest.approx(10.0)

    @staticmethod
    def test_otm_payoff():
        assert terminal_condition_call(90, 100) == pytest.approx(0.0)

    @staticmethod
    def test_atm_payoff():
        assert terminal_condition_call(100, 100) == pytest.approx(0.0)

    @staticmethod
    def test_matches_call_price_as_time_to_expiry_shrinks():
        near_expiry_price = call_price(110, 100, 0.05, 0.20, 1e-8)
        payoff = terminal_condition_call(110, 100)
        assert near_expiry_price == pytest.approx(payoff, abs=1e-4)

    @staticmethod
    def test_array_input():
        stock_prices = np.array([80.0, 100.0, 120.0])
        payoffs = terminal_condition_call(stock_prices, 100)
        np.testing.assert_allclose(payoffs, [0.0, 0.0, 20.0])

    @staticmethod
    def test_negative_stock_price_raises():
        with pytest.raises(ValueError):
            terminal_condition_call(-100, 100)

    @staticmethod
    def test_zero_strike_price_raises():
        with pytest.raises(ValueError):
            terminal_condition_call(100, 0)


class TestTerminalConditionPut:
    @staticmethod
    def test_itm_payoff():
        assert terminal_condition_put(90, 100) == pytest.approx(10.0)

    @staticmethod
    def test_otm_payoff():
        assert terminal_condition_put(110, 100) == pytest.approx(0.0)

    @staticmethod
    def test_atm_payoff():
        assert terminal_condition_put(100, 100) == pytest.approx(0.0)

    @staticmethod
    def test_matches_put_price_as_time_to_expiry_shrinks():
        near_expiry_price = put_price(90, 100, 0.05, 0.20, 1e-8)
        payoff = terminal_condition_put(90, 100)
        assert near_expiry_price == pytest.approx(payoff, abs=1e-4)

    @staticmethod
    def test_array_input():
        stock_prices = np.array([80.0, 100.0, 120.0])
        payoffs = terminal_condition_put(stock_prices, 100)
        np.testing.assert_allclose(payoffs, [20.0, 0.0, 0.0])

    @staticmethod
    def test_negative_stock_price_raises():
        with pytest.raises(ValueError):
            terminal_condition_put(-100, 100)


class TestCallBoundaryConditionLower:
    @staticmethod
    def test_returns_zero():
        assert call_boundary_condition_lower(1.0) == pytest.approx(0.0)

    @staticmethod
    def test_matches_call_price_at_tiny_stock_price():
        analytical_price = call_price(1e-6, 100, 0.05, 0.20, 1)
        boundary_value = call_boundary_condition_lower(1.0)
        assert analytical_price == pytest.approx(boundary_value, abs=1e-3)

    @staticmethod
    def test_array_input_preserves_shape():
        times = np.array([0.5, 1.0, 2.0])
        result = call_boundary_condition_lower(times)
        assert result.shape == times.shape
        np.testing.assert_allclose(result, np.zeros(3))

    @staticmethod
    def test_negative_time_to_expiry_raises():
        with pytest.raises(ValueError):
            call_boundary_condition_lower(-1.0)


class TestCallBoundaryConditionUpper:
    @staticmethod
    def test_matches_call_price_at_huge_stock_price():
        analytical_price = call_price(1e7, 100, 0.05, 0.20, 1)
        boundary_value = call_boundary_condition_upper(1e7, 100, 0.05, 1)
        assert analytical_price == pytest.approx(boundary_value, rel=1e-6)

    @staticmethod
    def test_reduces_to_forward_value_with_no_dividend():
        result = call_boundary_condition_upper(1e6, 100, 0.05, 1)
        expected = 1e6 - 100 * np.exp(-0.05 * 1)
        assert result == pytest.approx(expected, rel=1e-10)

    @staticmethod
    def test_negative_stock_price_raises():
        with pytest.raises(ValueError):
            call_boundary_condition_upper(-100, 100, 0.05, 1)

    @staticmethod
    def test_zero_strike_price_raises():
        with pytest.raises(ValueError):
            call_boundary_condition_upper(100, 0, 0.05, 1)


class TestPutBoundaryConditionLower:
    @staticmethod
    def test_equals_discounted_strike():
        result = put_boundary_condition_lower(100, 0.05, 1)
        expected = 100 * np.exp(-0.05 * 1)
        assert result == pytest.approx(expected, rel=1e-10)

    @staticmethod
    def test_matches_put_price_at_tiny_stock_price():
        analytical_price = put_price(1e-6, 100, 0.05, 0.20, 1)
        boundary_value = put_boundary_condition_lower(100, 0.05, 1)
        assert analytical_price == pytest.approx(boundary_value, abs=1e-3)

    @staticmethod
    def test_negative_strike_price_raises():
        with pytest.raises(ValueError):
            put_boundary_condition_lower(-100, 0.05, 1)


class TestPutBoundaryConditionUpper:
    @staticmethod
    def test_returns_zero():
        assert put_boundary_condition_upper(1e6) == pytest.approx(0.0)

    @staticmethod
    def test_matches_put_price_at_huge_stock_price():
        analytical_price = put_price(1e7, 100, 0.05, 0.20, 1)
        boundary_value = put_boundary_condition_upper(1e7)
        assert analytical_price == pytest.approx(boundary_value, abs=1e-6)

    @staticmethod
    def test_array_input_preserves_shape():
        stock_prices = np.array([1e5, 1e6, 1e7])
        result = put_boundary_condition_upper(stock_prices)
        assert result.shape == stock_prices.shape
        np.testing.assert_allclose(result, np.zeros(3))

    @staticmethod
    def test_negative_stock_price_raises():
        with pytest.raises(ValueError):
            put_boundary_condition_upper(-100)


class TestCrossConsistency:
    @staticmethod
    def test_call_satisfies_pde_and_correct_boundaries_simultaneously():
        S, K, r, sigma, T, q = 100, 100, 0.05, 0.20, 1.0, 0.0

        _, pde_satisfied = verify_black_scholes_pde("call", S, K, r, sigma, T, q)
        lower = call_boundary_condition_lower(T)
        upper = call_boundary_condition_upper(1e7, K, r, T, q)
        payoff_at_expiry = terminal_condition_call(S, K)
        near_expiry_price = call_price(S, K, r, sigma, 1e-8, q)

        assert pde_satisfied is True
        assert lower == pytest.approx(0.0)
        assert call_price(1e7, K, r, sigma, T, q) == pytest.approx(upper, rel=1e-6)
        assert near_expiry_price == pytest.approx(payoff_at_expiry, abs=1e-3)

    @staticmethod
    def test_put_satisfies_pde_and_correct_boundaries_simultaneously():
        S, K, r, sigma, T, q = 100, 100, 0.05, 0.20, 1.0, 0.0

        _, pde_satisfied = verify_black_scholes_pde("put", S, K, r, sigma, T, q)
        lower = put_boundary_condition_lower(K, r, T)
        upper = put_boundary_condition_upper(1e7)
        payoff_at_expiry = terminal_condition_put(S, K)
        near_expiry_price = put_price(S, K, r, sigma, 1e-8, q)

        assert pde_satisfied is True
        assert put_price(1e-6, K, r, sigma, T, q) == pytest.approx(lower, abs=1e-3)
        assert upper == pytest.approx(0.0)
        assert near_expiry_price == pytest.approx(payoff_at_expiry, abs=1e-3)
