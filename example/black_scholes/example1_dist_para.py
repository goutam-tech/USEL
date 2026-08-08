"""
Example usage of the Black-Scholes distributions and parameters modules.
"""

from usel.black_scholes.distributions import (
    inverse_normal_cdf,
    normal_cdf,
    normal_log_pdf,
    normal_pdf,
    normal_sf,
)
from usel.black_scholes.parameters import d1, d2


def main() -> None:
    print("=" * 60)
    print("STANDARD NORMAL DISTRIBUTION")
    print("=" * 60)

    x = 0.5

    print(f"x = {x}\n")

    print(f"PDF              : {normal_pdf(x):.10f}")
    print(f"Log PDF          : {normal_log_pdf(x):.10f}")
    print(f"CDF              : {normal_cdf(x):.10f}")
    print(f"Survival Function: {normal_sf(x):.10f}")
    print(f"Inverse CDF(0.95): {inverse_normal_cdf(0.95):.10f}")

    print()

    print("=" * 60)
    print("BLACK-SCHOLES PARAMETERS")
    print("=" * 60)

    stock_price = 100.0
    strike_price = 100.0
    risk_free_rate = 0.05
    volatility = 0.20
    time_to_expiry = 1.0
    dividend_yield = 0.0

    print(f"Stock Price      : {stock_price}")
    print(f"Strike Price     : {strike_price}")
    print(f"Risk-Free Rate   : {risk_free_rate}")
    print(f"Volatility       : {volatility}")
    print(f"Time to Expiry   : {time_to_expiry}")
    print(f"Dividend Yield   : {dividend_yield}")

    print()

    d1_value = d1(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    d2_value = d2(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiry,
        dividend_yield,
    )

    print(f"d1 = {d1_value:.10f}")
    print(f"d2 = {d2_value:.10f}")

    print()

    print("=" * 60)
    print("VERIFY RELATION")
    print("=" * 60)

    expected = d1_value - volatility * (time_to_expiry**0.5)

    print(f"d1 - σ√T = {expected:.10f}")
    print(f"d2       = {d2_value:.10f}")
    print(f"Equal?   = {abs(expected - d2_value) < 1e-12}")

    print()

    print("=" * 60)
    print("ARRAY EXAMPLE")
    print("=" * 60)

    stock_prices = [90, 100, 110, 120]

    print("Stock\t\tPDF\t\tCDF\t\td1\t\td2")

    for stock in stock_prices:
        d1_val = d1(
            stock,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )

        d2_val = d2(
            stock,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )

        pdf = normal_pdf(d1_val)
        cdf = normal_cdf(d1_val)

        print(f"{stock:<10}{pdf:.6f}\t{cdf:.6f}\t{d1_val:.6f}\t{d2_val:.6f}")


if __name__ == "__main__":
    main()
