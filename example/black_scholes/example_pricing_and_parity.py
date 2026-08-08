import numpy as np

from usel.black_scholes.parity import (
    call_from_put,
    parity_residual,
    put_from_call,
    verify_put_call_parity,
)
from usel.black_scholes.pricing import call_price, put_price

stock_price = 100
strike_price = 100
risk_free_rate = 0.05
volatility = 0.20
time_to_expiry = 1
dividend_yield = 0.0

call = call_price(stock_price, strike_price, risk_free_rate, volatility, time_to_expiry)
put = put_price(stock_price, strike_price, risk_free_rate, volatility, time_to_expiry)

print(f"Call price : {call:.6f}")
print(f"Put price  : {put:.6f}")

residual, satisfied = verify_put_call_parity(
    call, put, stock_price, strike_price, risk_free_rate, time_to_expiry
)

print(f"Parity residual : {residual:.2e}")
print(f"Parity satisfied: {satisfied}")

put_via_parity = put_from_call(call, stock_price, strike_price, risk_free_rate, time_to_expiry)
print(f"Put via parity: {put_via_parity:.6f} (direct: {put:.6f})")

call_via_parity = call_from_put(put, stock_price, strike_price, risk_free_rate, time_to_expiry)
print(f"Call via parity: {call_via_parity:.6f} (direct: {call:.6f})")

strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

calls = call_price(stock_price, strikes, risk_free_rate, volatility, time_to_expiry)
puts = put_price(stock_price, strikes, risk_free_rate, volatility, time_to_expiry)

residuals = parity_residual(calls, puts, stock_price, strikes, risk_free_rate, time_to_expiry)

print("\nStrike ladder parity check:")
for k, c, p, r in zip(strikes, calls, puts, residuals, strict=True):
    print(f"  K={k:>6.1f}  C={c:>8.4f}  P={p:>8.4f}  residual={r:.2e}")

_, satisfied_broken = verify_put_call_parity(
    call + 1.0, put, stock_price, strike_price, risk_free_rate, time_to_expiry
)
print(f"\nParity satisfied with tampered call price: {satisfied_broken}")
