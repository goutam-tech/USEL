"""Benchmark: matrix multiplication performance across sizes."""

from __future__ import annotations

import time

from usel.math import Matrix

SIZES = (50, 100, 200, 400)

def main() -> None:
    print(f"{'Size':>8} | {'Time (s)':>10}")
    print("-" * 23)
    for size in SIZES:
        a = Matrix.random(size, size, seed=1)
        b = Matrix.random(size, size, seed=2)
        start = time.perf_counter()
        _ = a @ b
        elapsed = time.perf_counter() - start
        print(f"{size:>8} | {elapsed:>10.6f}")


if __name__ == "__main__":
    main()