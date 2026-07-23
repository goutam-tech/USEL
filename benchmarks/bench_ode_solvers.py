"""Benchmark: ODE solver performance comparison."""

from __future__ import annotations

import time

from usel.solvers import euler, improved_euler, rk2, rk4


def decay(t: float, y: float) -> float:
    return -y


def main() -> None:
    methods = {"euler": euler, "improved_euler": improved_euler, "rk2": rk2, "rk4": rk4}
    print(f"{'Method':>16} | {'Time (s)':>10}")
    print("-" * 31)
    for name, method in methods.items():
        start = time.perf_counter()
        method(decay, t0=0.0, y0=1.0, t_end=100.0, h=0.001)
        elapsed = time.perf_counter() - start
        print(f"{name:>16} | {elapsed:>10.6f}")


if __name__ == "__main__":
    main()
