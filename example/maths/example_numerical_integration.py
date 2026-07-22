"""Example 4: Numerical Integration — comparing Trapezoidal, Simpson, and
Gaussian Quadrature against a known analytic result.
"""

from __future__ import annotations
import math
from usel.solvers import gaussian_quadrature, simpson, trapezoidal

def main() -> None:
    result_trap = trapezoidal(math.sin, 0.0, math.pi, n=1000)
    result_simpson = simpson(math.sin, 0.0, math.pi, n=1000)
    result_gauss = gaussian_quadrature(math.sin, 0.0, math.pi, n=10)

    print("Integral of sin(x) from 0 to pi (exact = 2.0):")
    print(f"  Trapezoidal:        {result_trap:.8f}")
    print(f"  Simpson:            {result_simpson:.8f}")
    print(f"  Gaussian Quadrature:{result_gauss:.8f}")


if __name__ == "__main__":
    main()