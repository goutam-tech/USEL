"""
Tests for black_scholes.stochastic
"""

from __future__ import annotations

import numpy as np
import pytest

from usel.black_scholes.pricing import call_price, put_price
from usel.black_scholes.stochastic import (
    MonteCarloResult,
    monte_carlo_call_price,
    monte_carlo_price,
    monte_carlo_put_price,
    simulate_gbm_paths,
    simulate_gbm_terminal,
)


class TestSimulateGbmTerminal:
    def test_output_shape(self):
        terminal = simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=10_000, random_seed=0)
        assert terminal.shape == (10_000,)

    def test_all_prices_positive(self):
        terminal = simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=10_000, random_seed=0)
        assert np.all(terminal > 0.0)

    def test_mean_matches_risk_neutral_expectation(self):
        stock_price, risk_free_rate, volatility, time_to_expiry = 100, 0.05, 0.20, 1
        terminal = simulate_gbm_terminal(
            stock_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_paths=500_000,
            random_seed=1,
        )
        expected_mean = stock_price * np.exp(risk_free_rate * time_to_expiry)
        assert np.mean(terminal) == pytest.approx(expected_mean, rel=1e-2)

    def test_mean_reflects_dividend_yield(self):
        stock_price, risk_free_rate, volatility, time_to_expiry, q = (
            100,
            0.05,
            0.20,
            1,
            0.03,
        )
        terminal = simulate_gbm_terminal(
            stock_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            q,
            num_paths=500_000,
            random_seed=1,
        )
        expected_mean = stock_price * np.exp((risk_free_rate - q) * time_to_expiry)
        assert np.mean(terminal) == pytest.approx(expected_mean, rel=1e-2)

    def test_reproducible_with_same_seed(self):
        first = simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=1000, random_seed=7)
        second = simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=1000, random_seed=7)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_results(self):
        first = simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=1000, random_seed=1)
        second = simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=1000, random_seed=2)
        assert not np.array_equal(first, second)

    def test_zero_time_to_expiry_returns_constant(self):
        terminal = simulate_gbm_terminal(100, 0.05, 0.20, 0, num_paths=1000, random_seed=0)
        np.testing.assert_allclose(terminal, np.full(1000, 100.0))

    def test_antithetic_pairs_present(self):
        terminal = simulate_gbm_terminal(
            100, 0.05, 0.20, 1, num_paths=1000, antithetic=True, random_seed=3
        )
        log_returns = np.log(terminal / 100.0)
        half = 500
        drift = (0.05 - 0.5 * 0.20**2) * 1
        diffusion_first = log_returns[:half] - drift
        diffusion_second = log_returns[half:] - drift
        np.testing.assert_allclose(diffusion_first, -diffusion_second, atol=1e-10)

    def test_negative_stock_price_raises(self):
        with pytest.raises(ValueError):
            simulate_gbm_terminal(-100, 0.05, 0.20, 1)

    def test_negative_volatility_raises(self):
        with pytest.raises(ValueError):
            simulate_gbm_terminal(100, 0.05, -0.20, 1)

    def test_negative_time_to_expiry_raises(self):
        with pytest.raises(ValueError):
            simulate_gbm_terminal(100, 0.05, 0.20, -1)

    def test_too_few_paths_raises(self):
        with pytest.raises(ValueError):
            simulate_gbm_terminal(100, 0.05, 0.20, 1, num_paths=1)


class TestSimulateGbmPaths:
    def test_output_shape(self):
        paths = simulate_gbm_paths(100, 0.05, 0.20, 1, num_paths=500, num_steps=50, random_seed=0)
        assert paths.shape == (500, 51)

    def test_first_column_equals_initial_price(self):
        paths = simulate_gbm_paths(100, 0.05, 0.20, 1, num_paths=500, num_steps=50, random_seed=0)
        np.testing.assert_allclose(paths[:, 0], np.full(500, 100.0))

    def test_all_prices_positive(self):
        paths = simulate_gbm_paths(100, 0.05, 0.20, 1, num_paths=500, num_steps=50, random_seed=0)
        assert np.all(paths > 0.0)

    def test_terminal_column_mean_matches_expectation(self):
        stock_price, risk_free_rate, volatility, time_to_expiry = 100, 0.05, 0.20, 1
        paths = simulate_gbm_paths(
            stock_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_paths=50_000,
            num_steps=100,
            random_seed=2,
        )
        expected_mean = stock_price * np.exp(risk_free_rate * time_to_expiry)
        assert np.mean(paths[:, -1]) == pytest.approx(expected_mean, rel=2e-2)

    def test_too_few_steps_raises(self):
        with pytest.raises(ValueError):
            simulate_gbm_paths(100, 0.05, 0.20, 1, num_steps=0)


class TestMonteCarloCallPrice:
    def test_returns_named_tuple(self):
        result = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=10_000, random_seed=0)
        assert isinstance(result, MonteCarloResult)

    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry",
        [
            (100, 100, 0.05, 0.20, 1.0),
            (80, 100, 0.03, 0.35, 0.5),
            (120, 90, 0.01, 0.15, 2.0),
        ],
    )
    def test_price_within_confidence_interval_of_analytical(
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
        result = monte_carlo_call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_paths=200_000,
            random_seed=0,
        )
        assert result.conf_interval_lower <= analytical <= result.conf_interval_upper

    def test_price_close_to_analytical_with_dividend_yield(self):
        analytical = call_price(100, 100, 0.05, 0.20, 1, 0.03)
        result = monte_carlo_call_price(
            100, 100, 0.05, 0.20, 1, 0.03, num_paths=200_000, random_seed=0
        )
        assert result.price == pytest.approx(analytical, abs=4 * result.std_error)

    def test_std_error_positive(self):
        result = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=10_000, random_seed=0)
        assert result.std_error > 0.0

    def test_conf_interval_brackets_price(self):
        result = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=10_000, random_seed=0)
        assert result.conf_interval_lower < result.price < result.conf_interval_upper

    def test_num_paths_reported_correctly(self):
        result = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=12_345, random_seed=0)
        assert result.num_paths == 12_345


class TestMonteCarloPutPrice:
    @pytest.mark.parametrize(
        "stock_price,strike_price,risk_free_rate,volatility,time_to_expiry",
        [
            (100, 100, 0.05, 0.20, 1.0),
            (80, 100, 0.03, 0.35, 0.5),
            (120, 90, 0.01, 0.15, 2.0),
        ],
    )
    def test_price_within_confidence_interval_of_analytical(
        self,
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
        result = monte_carlo_put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_paths=200_000,
            random_seed=0,
        )
        assert result.conf_interval_lower <= analytical <= result.conf_interval_upper


class TestMonteCarloPriceDispatcher:
    def test_dispatches_to_call(self):
        analytical = call_price(100, 100, 0.05, 0.20, 1)
        result = monte_carlo_price(
            "call", 100, 100, 0.05, 0.20, 1, num_paths=200_000, random_seed=0
        )
        assert result.conf_interval_lower <= analytical <= result.conf_interval_upper

    def test_dispatches_to_put(self):
        analytical = put_price(100, 100, 0.05, 0.20, 1)
        result = monte_carlo_price("put", 100, 100, 0.05, 0.20, 1, num_paths=200_000, random_seed=0)
        assert result.conf_interval_lower <= analytical <= result.conf_interval_upper

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_price("straddle", 100, 100, 0.05, 0.20, 1)


class TestAntitheticVarianceReduction:
    def test_antithetic_reduces_std_error(self):
        with_antithetic = monte_carlo_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_paths=20_000,
            antithetic=True,
            random_seed=0,
        )
        without_antithetic = monte_carlo_call_price(
            100,
            100,
            0.05,
            0.20,
            1,
            num_paths=20_000,
            antithetic=False,
            random_seed=0,
        )
        assert with_antithetic.std_error < without_antithetic.std_error


class TestConvergence:
    def test_std_error_shrinks_with_more_paths(self):
        small = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=1_000, random_seed=0)
        large = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=100_000, random_seed=0)
        assert large.std_error < small.std_error

    def test_price_converges_to_analytical_with_many_paths(self):
        analytical = call_price(100, 100, 0.05, 0.20, 1)
        result = monte_carlo_call_price(100, 100, 0.05, 0.20, 1, num_paths=500_000, random_seed=0)
        assert result.price == pytest.approx(analytical, abs=0.05)


class TestZeroTimeToExpiry:
    def test_call_price_equals_intrinsic(self):
        result = monte_carlo_call_price(110, 100, 0.05, 0.20, 0, num_paths=1000)
        assert result.price == pytest.approx(10.0, abs=1e-8)
        assert result.std_error == pytest.approx(0.0, abs=1e-8)

    def test_put_price_equals_intrinsic(self):
        result = monte_carlo_put_price(90, 100, 0.05, 0.20, 0, num_paths=1000)
        assert result.price == pytest.approx(10.0, abs=1e-8)


class TestValidation:
    def test_negative_stock_price_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_call_price(-100, 100, 0.05, 0.20, 1)

    def test_zero_strike_price_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_call_price(100, 0, 0.05, 0.20, 1)

    def test_negative_time_to_expiry_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_call_price(100, 100, 0.05, 0.20, -1)

    def test_invalid_confidence_level_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_call_price(100, 100, 0.05, 0.20, 1, confidence_level=1.5)

    def test_zero_confidence_level_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_call_price(100, 100, 0.05, 0.20, 1, confidence_level=0.0)


class TestMonteCarloPutCallParity:
    def test_parity_holds_within_combined_uncertainty(self):
        stock_price, strike_price, risk_free_rate, volatility, time_to_expiry = (
            100,
            100,
            0.05,
            0.20,
            1,
        )

        call_result = monte_carlo_call_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_paths=200_000,
            random_seed=0,
        )
        put_result = monte_carlo_put_price(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_paths=200_000,
            random_seed=0,
        )

        lhs = call_result.price - put_result.price
        rhs = stock_price - strike_price * np.exp(-risk_free_rate * time_to_expiry)

        combined_std_error = np.sqrt(call_result.std_error**2 + put_result.std_error**2)

        assert lhs == pytest.approx(rhs, abs=4 * combined_std_error)
