"""
example_greeks_and_implied_volatility.py

Demonstrates black_scholes.greeks and black_scholes.implied_volatility
working together: start from an observed market price, solve for the
implied volatility, then compute the full Greek profile at that
implied volatility -- exactly the workflow a trading desk uses to go
from quoted option prices to risk sensitivities.
"""

import numpy as np

from usel.black_scholes.greeks import (
    delta_call,
    gamma,
    rho_call,
    theta_call,
    vega,
)
from usel.black_scholes.implied_volatility import implied_volatility_call
from usel.black_scholes.pricing import call_price

market_price = 12.50
stock_price = 105.0
strike_price = 100.0
risk_free_rate = 0.04
time_to_expiry = 0.5
dividend_yield = 0.01

iv = implied_volatility_call(
    market_price,
    stock_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    dividend_yield,
)

print(f"Market price      : {market_price:.4f}")
print(f"Implied volatility : {iv:.4%}")

repriced = call_price(
    stock_price,
    strike_price,
    risk_free_rate,
    iv,
    time_to_expiry,
    dividend_yield,
)
print(f"Re-priced at IV    : {repriced:.4f} (diff: {repriced - market_price:.2e})")

d = delta_call(
    stock_price,
    strike_price,
    risk_free_rate,
    iv,
    time_to_expiry,
    dividend_yield,
)
g = gamma(
    stock_price,
    strike_price,
    risk_free_rate,
    iv,
    time_to_expiry,
    dividend_yield,
)
v = vega(
    stock_price,
    strike_price,
    risk_free_rate,
    iv,
    time_to_expiry,
    dividend_yield,
)
t = theta_call(
    stock_price,
    strike_price,
    risk_free_rate,
    iv,
    time_to_expiry,
    dividend_yield,
)
r = rho_call(
    stock_price,
    strike_price,
    risk_free_rate,
    iv,
    time_to_expiry,
    dividend_yield,
)

print("\nGreeks at implied volatility:")
print(f"  Delta : {d:.4f}")
print(f"  Gamma : {g:.6f}")
print(f"  Vega  : {v:.4f}  (per 100 vol pts; ${v / 100:.4f} per 1 vol pt)")
print(f"  Theta : {t:.4f}  (per year; ${t / 365:.4f} per day)")
print(f"  Rho   : {r:.4f}  (per 100 rate pts; ${r / 100:.4f} per 1bp*100)")

strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])

true_skewed_vols = 0.25 - 0.001 * (strikes - 100.0)
market_prices = call_price(
    stock_price,
    strikes,
    risk_free_rate,
    true_skewed_vols,
    time_to_expiry,
    dividend_yield,
)

implied_vols = implied_volatility_call(
    market_prices,
    stock_price,
    strikes,
    risk_free_rate,
    time_to_expiry,
    dividend_yield,
)

vegas = vega(
    stock_price,
    strikes,
    risk_free_rate,
    implied_vols,
    time_to_expiry,
    dividend_yield,
)

print("\nImplied volatility smile:")
for k, iv_k, vega_k in zip(strikes, implied_vols, vegas, strict=True):
    print(f"  K={k:>6.1f}  IV={iv_k:.4%}  Vega={vega_k:.4f}")

total_vega_exposure = np.sum(vegas)
print(f"\nTotal portfolio vega across strikes: {total_vega_exposure:.4f}")
