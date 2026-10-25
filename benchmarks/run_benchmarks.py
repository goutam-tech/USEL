"""Run the reproducible CPU and memory benchmark suite.

Usage
-----
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --repeats 10 --json results/bench.json
    python benchmarks/run_benchmarks.py --quick          # small, fast sizes

All inputs use fixed seeds, so repeated runs on the same machine measure the
same work. Setup cost is excluded from timings.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from usel.math import Matrix, fft
from usel.solvers.ode import rk4
from usel.utils.benchmarking import (
    BenchmarkResult,
    environment_info,
    format_table,
    run_benchmark,
    write_json,
)

SEED_A = 1
SEED_B = 2


def _decay(_t: float, y: float) -> float:
    return -y


def build_suite(
    quick: bool, repeats: int, warmup: int, track_memory: bool
) -> list[BenchmarkResult]:
    """Run every benchmark case and return the results."""
    matmul_sizes = (50, 100) if quick else (50, 100, 200, 400)
    fft_sizes = (2**10, 2**12) if quick else (2**10, 2**14, 2**18)
    ode_end = 10.0 if quick else 100.0

    common = {"repeats": repeats, "warmup": warmup, "track_memory": track_memory}
    results: list[BenchmarkResult] = []

    for n in matmul_sizes:

        def setup_matmul(n: int = n) -> tuple[Matrix, Matrix]:
            return Matrix.random(n, n, seed=SEED_A), Matrix.random(n, n, seed=SEED_B)

        results.append(
            run_benchmark(
                "matmul",
                lambda a, b: a @ b,
                setup=setup_matmul,
                params={"size": n},
                **common,
            )
        )

    for n in fft_sizes:

        def setup_fft(n: int = n) -> tuple[np.ndarray]:
            return (np.random.default_rng(SEED_A).random(n),)

        results.append(run_benchmark("fft", fft, setup=setup_fft, params={"n": n}, **common))

    results.append(
        run_benchmark(
            "ode_rk4",
            lambda: rk4(_decay, t0=0.0, y0=1.0, t_end=ode_end, h=0.001),
            params={"t_end": ode_end, "h": 0.001},
            **common,
        )
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="USEL CPU and memory benchmarks")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per case (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up runs (default 1)")
    parser.add_argument("--quick", action="store_true", help="use small sizes for a fast run")
    parser.add_argument("--no-memory", action="store_true", help="skip peak-memory measurement")
    parser.add_argument("--json", metavar="PATH", help="also write results to this JSON file")
    args = parser.parse_args(argv)

    env = environment_info()
    print("Environment:")
    for key, value in env.items():
        print(f"  {key:<14} {value}")
    print()

    results = build_suite(args.quick, args.repeats, args.warmup, not args.no_memory)
    print(format_table(results))

    if args.json:
        path = write_json(results, args.json)
        print(f"\nResults written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
