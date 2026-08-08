"""
example_three_way_validation.py

Prices a single European option three independent ways and confirms
they agree:

1. pde.py             -- verifies the closed-form price satisfies the
                          Black-Scholes PDE exactly (analytical check)
2. finite_difference.py -- solves the PDE numerically on a grid
3. stochastic.py       -- Monte Carlo simulation of the risk-neutral
                          GBM process

If a bug were introduced into any one implementation, this script
would very likely catch it, since three structurally different
methods would stop agreeing.
"""

import numpy as np

from usel.black_scholes.finite_difference import (
    finite_difference_call_price,
    solve_black_scholes_pde,
)
from usel.black_scholes.pde import verify_black_scholes_pde
from usel.black_scholes.pricing import call_price
from usel.black_scholes.stochastic import monte_carlo_call_price

stock_price = 100.0
strike_price = 100.0
risk_free_rate = 0.05
volatility = 0.20
time_to_expiry = 1.0
dividend_yield = 0.0

analytical_price = call_price(
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
)

pde_residual, pde_satisfied = verify_black_scholes_pde(
    "call",
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
)

print("Method 1: Closed-form (pricing.py + pde.py)")
print(f"  Price          : {analytical_price:.6f}")
print(f"  PDE residual   : {pde_residual:.2e}")
print(f"  PDE satisfied  : {pde_satisfied}")

fd_price = finite_difference_call_price(
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
    num_space_steps=300,
    num_time_steps=300,
)

fd_error = abs(fd_price - analytical_price)

print("\nMethod 2: Finite-difference PDE solver (finite_difference.py)")
print(f"  Price          : {fd_price:.6f}")
print(f"  Abs. error     : {fd_error:.6f}")


mc_result = monte_carlo_call_price(
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
    num_paths=500_000,
    random_seed=42,
)

print("\nMethod 3: Monte Carlo simulation (stochastic.py)")
print(f"  Price          : {mc_result.price:.6f}")
print(f"  Std. error     : {mc_result.std_error:.6f}")
print(
    f"  95% CI         : [{mc_result.conf_interval_lower:.6f}, {mc_result.conf_interval_upper:.6f}]"
)

print("\nCross-validation:")
print(f"  Closed-form vs finite-difference agree within 0.01: {fd_error < 0.01}")
mc_agrees = mc_result.conf_interval_lower <= analytical_price <= mc_result.conf_interval_upper
print(f"  Closed-form falls inside Monte Carlo 95% CI: {mc_agrees}")

grid = solve_black_scholes_pde(
    "call",
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
    num_space_steps=300,
    num_time_steps=300,
)

sample_stock_prices = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
grid_prices = np.interp(sample_stock_prices, grid.stock_grid, grid.price_grid)
analytical_prices = call_price(
    sample_stock_prices,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiry,
    dividend_yield,
)

print("\nFinite-difference grid vs. closed-form across strikes:")
for s, g, a in zip(sample_stock_prices, grid_prices, analytical_prices, strict=True):
    print(f"  S={s:>6.1f}  FD={g:>9.4f}  Analytical={a:>9.4f}  diff={abs(g - a):.4f}")
