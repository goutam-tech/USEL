"""Example 5: ODE Solver — exponential decay via RK4."""

from __future__ import annotations
import math
from usel.solvers import rk4

def decay(t: float, y: float) -> float:
    return -y

def main() -> None:
    t, y = rk4(decay, t0=0.0, y0=1.0, t_end=5.0, h=0.1)
    print("t\t y (RK4)\t y (exact)")
    for i in range(0, len(t), 10):
        exact = math.exp(-t[i])
        print(f"{t[i]:.2f}\t {y[i]:.6f}\t {exact:.6f}")


if __name__ == "__main__":
    main()