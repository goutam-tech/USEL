"""
Tests for black_scholes.finite_difference
"""

from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.finite_difference import (
    FiniteDifferenceGrid,
    finite_difference_call_price,
    finite_difference_price,
    finite_difference_put_price,
    solve_black_scholes_pde,
)
from usel.black_scholes.pricing import call_price, put_price


class TestCrankNicolsonAccuracy:
    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry",
        [
            (100, 100, 0.05, 0.20, 1.0),
            (80, 100, 0.03, 0.35, 0.5),
            (120, 90, 0.01, 0.15, 2.0),
            (50, 60, 0.10, 0.40, 0.25),
        ],
    )
    def test_call_matches_closed_form(
        self,
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
    ):
        analytical = call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )
        numerical = finite_difference_call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )
        assert numerical == pytest.approx(analytical, abs=5e-2)

    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry",
        [
            (100, 100, 0.05, 0.20, 1.0),
            (80, 100, 0.03, 0.35, 0.5),
            (120, 90, 0.01, 0.15, 2.0),
            (50, 60, 0.10, 0.40, 0.25),
        ],
    )
    @staticmethod
    def test_put_matches_closed_form(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
    ):
        analytical = put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )
        numerical = finite_difference_put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )
        assert numerical == pytest.approx(analytical, abs=5e-2)

    @staticmethod
    def test_call_matches_closed_form_with_dividend_yield():
        analytical = call_price(100, 100, 0.05, 0.20, 1, 0.03)
        numerical = finite_difference_call_price(100, 100, 0.05, 0.20, 1, 0.03)
        assert numerical == pytest.approx(analytical, abs=5e-2)


class TestImplicitAccuracy:
    @staticmethod
    def test_call_matches_closed_form():
        analytical = call_price(100, 100, 0.05, 0.20, 1)
        numerical = finite_difference_call_price(100, 100, 0.05, 0.20, 1, scheme="implicit")
        assert numerical == pytest.approx(analytical, abs=1e-1)

    @staticmethod
    def test_put_matches_closed_form():
        analytical = put_price(100, 100, 0.05, 0.20, 1)
        numerical = finite_difference_put_price(100, 100, 0.05, 0.20, 1, scheme="implicit")
        assert numerical == pytest.approx(analytical, abs=1e-1)


class TestExplicitAccuracy:
    @staticmethod
    def test_call_matches_closed_form_with_stable_grid():
        analytical = call_price(100, 100, 0.05, 0.20, 1)
        numerical = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=100,
            num_time_steps=3000,
            scheme="explicit",
        )
        assert numerical == pytest.approx(analytical, abs=1e-1)


class TestConvergence:
    @staticmethod
    def test_error_decreases_with_finer_grid():
        analytical = call_price(100, 100, 0.05, 0.20, 1)

        coarse = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=25,
            num_time_steps=25,
        )
        fine = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=200,
            num_time_steps=200,
        )

        coarse_error = abs(coarse - analytical)
        fine_error = abs(fine - analytical)

        assert fine_error < coarse_error

    @staticmethod
    def test_fine_grid_within_tight_tolerance():
        analytical = call_price(100, 100, 0.05, 0.20, 1)
        numerical = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=400,
            num_time_steps=400,
        )
        assert numerical == pytest.approx(analytical, abs=1e-2)


class TestZeroTimeToExpiry:
    @staticmethod
    def test_call_grid_equals_payoff():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 0)
        expected_payoff = np.maximum(grid.stock_grid - 100, 0.0)
        np.testing.assert_allclose(grid.price_grid, expected_payoff)

    @staticmethod
    def test_put_grid_equals_payoff():
        grid = solve_black_scholes_pde("put", 100, 0.05, 0.20, 0)
        expected_payoff = np.maximum(100 - grid.stock_grid, 0.0)
        np.testing.assert_allclose(grid.price_grid, expected_payoff)

    @staticmethod
    def test_finite_difference_call_price_equals_intrinsic():
        price = finite_difference_call_price(110, 100, 0.05, 0.20, 0)
        assert price == pytest.approx(10.0, abs=1e-6)


class TestGridStructure:
    @staticmethod
    def test_returns_named_tuple():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1)
        assert isinstance(grid, FiniteDifferenceGrid)

    @staticmethod
    def test_grid_length_matches_num_space_steps():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, num_space_steps=50)
        assert grid.stock_grid.shape == (51,)
        assert grid.price_grid.shape == (51,)

    @staticmethod
    def test_stock_grid_starts_at_zero():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1)
        assert grid.stock_grid[0] == pytest.approx(0.0)

    @staticmethod
    def test_stock_grid_ends_at_stock_price_max():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, stock_price_max=500.0)
        assert grid.stock_grid[-1] == pytest.approx(500.0)

    @staticmethod
    def test_call_price_non_negative_across_grid():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1)
        assert np.all(grid.price_grid >= -1e-8)

    @staticmethod
    def test_put_price_non_negative_across_grid():
        grid = solve_black_scholes_pde("put", 100, 0.05, 0.20, 1)
        assert np.all(grid.price_grid >= -1e-8)

    @staticmethod
    def test_call_price_monotonic_increasing_in_stock_price():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1)
        assert np.all(np.diff(grid.price_grid) >= -1e-8)

    @staticmethod
    def test_put_price_monotonic_decreasing_in_stock_price():
        grid = solve_black_scholes_pde("put", 100, 0.05, 0.20, 1)
        assert np.all(np.diff(grid.price_grid) <= 1e-8)


class TestBoundaryConditions:
    @staticmethod
    def test_call_lower_boundary_is_zero():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1)
        assert grid.price_grid[0] == pytest.approx(0.0, abs=1e-8)

    @staticmethod
    def test_call_upper_boundary_matches_forward_value():
        grid = solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, stock_price_max=800.0)
        expected = 800.0 - 100 * np.exp(-0.05 * 1)
        assert grid.price_grid[-1] == pytest.approx(expected, rel=1e-8)

    @staticmethod
    def test_put_upper_boundary_is_zero():
        grid = solve_black_scholes_pde("put", 100, 0.05, 0.20, 1)
        assert grid.price_grid[-1] == pytest.approx(0.0, abs=1e-8)

    @staticmethod
    def test_put_lower_boundary_matches_discounted_strike():
        grid = solve_black_scholes_pde("put", 100, 0.05, 0.20, 1)
        expected = 100 * np.exp(-0.05 * 1)
        assert grid.price_grid[0] == pytest.approx(expected, rel=1e-8)


class TestValidation:
    @staticmethod
    def test_invalid_option_type_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("straddle", 100, 0.05, 0.20, 1)

    @staticmethod
    def test_invalid_scheme_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, scheme="rk4")

    @staticmethod
    def test_zero_strike_price_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 0, 0.05, 0.20, 1)

    @staticmethod
    def test_negative_volatility_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 100, 0.05, -0.20, 1)

    @staticmethod
    def test_negative_time_to_expiry_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 100, 0.05, 0.20, -1)

    @staticmethod
    def test_too_few_space_steps_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, num_space_steps=1)

    @staticmethod
    def test_too_few_time_steps_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, num_time_steps=0)

    @staticmethod
    def test_negative_stock_price_max_raises():
        with pytest.raises(ValueError):
            solve_black_scholes_pde("call", 100, 0.05, 0.20, 1, stock_price_max=-10.0)

    @staticmethod
    def test_stock_price_outside_grid_raises():
        with pytest.raises(ValueError):
            finite_difference_call_price(1e9, 100, 0.05, 0.20, 1, stock_price_max=500.0)

    @staticmethod
    def test_finite_difference_price_invalid_option_type_raises():
        with pytest.raises(ValueError):
            finite_difference_price("straddle", 100, 100, 0.05, 0.20, 1)


class TestExplicitStabilityWarning:
    @staticmethod
    def test_warns_when_time_steps_too_few():
        with pytest.warns(RuntimeWarning):
            solve_black_scholes_pde(
                "call",
                100,
                0.05,
                0.20,
                1,
                num_space_steps=100,
                num_time_steps=5,
                scheme="explicit",
            )


class TestFiniteDifferencePriceDispatcher:
    @staticmethod
    def test_dispatches_to_call():
        analytical = call_price(100, 100, 0.05, 0.20, 1)
        numerical = finite_difference_price("call", 100, 100, 0.05, 0.20, 1)
        assert numerical == pytest.approx(analytical, abs=5e-2)

    @staticmethod
    def test_dispatches_to_put():
        analytical = put_price(100, 100, 0.05, 0.20, 1)
        numerical = finite_difference_price("put", 100, 100, 0.05, 0.20, 1)
        assert numerical == pytest.approx(analytical, abs=5e-2)

    @staticmethod
    def test_case_insensitive():
        numerical_lower = finite_difference_price("call", 100, 100, 0.05, 0.20, 1)
        numerical_upper = finite_difference_price("CALL", 100, 100, 0.05, 0.20, 1)
        assert numerical_lower == pytest.approx(numerical_upper, abs=1e-10)


class TestBroadcasting:
    @staticmethod
    def test_accepts_array_of_stock_prices():
        stock_prices = np.array([80.0, 100.0, 120.0])
        result = finite_difference_call_price(stock_prices, 100, 0.05, 0.20, 1)
        assert isinstance(result, np.ndarray)
        assert result.shape == stock_prices.shape

    @staticmethod
    def test_array_result_matches_analytical_elementwise():
        stock_prices = np.array([80.0, 100.0, 120.0])
        numerical = finite_difference_call_price(stock_prices, 100, 0.05, 0.20, 1)
        analytical = call_price(stock_prices, 100, 0.05, 0.20, 1)
        np.testing.assert_allclose(numerical, analytical, atol=5e-2)


class TestSchemeCrossConsistency:
    @staticmethod
    def test_all_schemes_roughly_agree_on_fine_grid():
        cn = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=200,
            num_time_steps=2000,
            scheme="crank-nicolson",
        )
        implicit = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=200,
            num_time_steps=2000,
            scheme="implicit",
        )
        explicit = finite_difference_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_space_steps=200,
            num_time_steps=6000,
            scheme="explicit",
        )

        assert cn == pytest.approx(implicit, abs=1e-1)
        assert cn == pytest.approx(explicit, abs=1e-1)
