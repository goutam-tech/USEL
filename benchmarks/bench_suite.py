"""Reproducible CPU and memory benchmarks for USEL's implemented methods.

Run locally on a standard CPU-only machine:

    uv run python benchmarks/bench_suite.py --out benchmark_results

Measures elapsed time across repeated runs (median + spread) and peak memory
using ``tracemalloc`` (Python standard library). ``tracemalloc`` tracks Python
allocations only; it does not capture raw numeric buffers allocated by C
extensions (e.g. some NumPy/SciPy internals), so reported memory is a lower
bound on total process memory — not a precise RSS measurement.

Results are exported to ``results.json`` and ``results.csv`` alongside a
readable summary. No GPU, network access, or secrets are required.

Numbers depend on hardware, Python/dependency versions, and system load. No
performance claim is made unless directly supported by a recorded, reproducible
measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from usel.black_scholes import call_price
from usel.linalg import svd
from usel.math import Matrix, fft
from usel.solvers import newton, rk4, simpson

REPEATS = 5
WARMUP_RUNS = 1

RESULTS_DIR = Path("benchmark_results")


def environment() -> dict:
    """Environment metadata for interpreting and reproducing results."""
    return {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def measure(fn, repeats: int = REPEATS, warmup: int = WARMUP_RUNS):
    """Run ``fn`` ``repeats`` times after ``warmup`` runs.

    Returns ``(median_seconds, stdev_seconds, peak_memory_mb)``. Callers keep
    setup and data generation outside ``fn`` so only the operation is timed;
    where that cannot be separated it is noted in the workload description.
    """
    for _ in range(warmup):
        fn()
    times = []
    tracemalloc.start()
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return median, stdev, peak_bytes / (1024 * 1024)


# ----------------------------------------------------------------------
# Representative workloads (implemented functionality only).
# Each entry: name, parameters recorded in output, short "why", and a
# zero-argument callable that runs the operation (setup already done).
# ----------------------------------------------------------------------
def workload_matrix_multiply():
    a = Matrix.random(200, 200, seed=1)
    b = Matrix.random(200, 200, seed=2)

    def run():
        return a @ b

    return (
        run,
        {"size": "200x200", "seed_a": 1, "seed_b": 2},
        "Dense linear-algebra core; O(n^3) multiply exercised by every decomposition path.",
    )


def workload_fft():
    signal = np.random.default_rng(7).standard_normal(8192)

    def run():
        return fft(signal)

    return (
        run,
        {"n": 8192, "seed": 7},
        "Spectral primitive used by split-operator propagators; fixed seed "
        "keeps input identical across runs.",
    )


def workload_ode_rk4():
    def decay(_t, y):
        return -y

    def run():
        return rk4(decay, t0=0.0, y0=1.0, t_end=10.0, h=0.001)

    return (
        run,
        {"method": "rk4", "t_end": 10.0, "h": 0.001},
        "Representative ODE integrator; smaller h than the doctest to make runtime measurable.",
    )


def workload_integration_simpson():
    def f(x):
        return x * x

    def run():
        return simpson(f, 0.0, 1.0, n=10_000)

    return (
        run,
        {"method": "simpson", "a": 0.0, "b": 1.0, "n": 10_000},
        "Numerical quadrature with an even subinterval count; deterministic, no random input.",
    )


def workload_root_newton():
    def f(x):
        return x * x - 2.0

    def df(x):
        return 2.0 * x

    def run():
        return newton(f, df, x0=1.0)

    return (
        run,
        {"method": "newton", "target": "sqrt(2)", "x0": 1.0},
        "Root finding with a known analytical answer; fixed derivative keeps iterations stable.",
    )


def workload_black_scholes_call():
    def run():
        return call_price(100.0, 100.0, 0.05, 0.20, 1.0)

    return (
        run,
        {"spot": 100.0, "strike": 100.0, "rate": 0.05, "vol": 0.20, "t": 1.0},
        "Quantitative-finance pricing path; analytical Black-Scholes with no dividend yield.",
    )


def workload_svd():
    a = Matrix.random(100, 100, seed=3)

    def run():
        return svd(a)

    return (
        run,
        {"size": "100x100", "seed": 3},
        "Matrix decomposition shared by linalg and eigensolver code paths.",
    )


WORKLOADS = [
    ("matrix_multiply", workload_matrix_multiply),
    ("fft", workload_fft),
    ("ode_rk4", workload_ode_rk4),
    ("integration_simpson", workload_integration_simpson),
    ("root_newton", workload_root_newton),
    ("black_scholes_call", workload_black_scholes_call),
    ("svd", workload_svd),
]


def run_benchmark(name: str | None = None, repeats: int = REPEATS) -> list[dict]:
    """Run every workload (or just ``name``) and return result records."""
    results = []
    for key, builder in WORKLOADS:
        if name is not None and key != name:
            continue
        run, params, why = builder()
        median, stdev, peak_mb = measure(run, repeats=repeats)
        results.append(
            {
                "name": key,
                "workload": why,
                "parameters": params,
                "repeats": repeats,
                "median_seconds": median,
                "stdev_seconds": stdev,
                "peak_memory_mb": peak_mb,
            }
        )
    return results


def write_outputs(results: list[dict], env: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"environment": env, "benchmarks": results}

    with (out_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "repeats",
                "median_seconds",
                "stdev_seconds",
                "peak_memory_mb",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'results.csv'}")


def print_summary(results: list[dict]) -> None:
    print(f"{'Workload':<24} {'median(s)':>12} {'stdev(s)':>12} {'peak_mem(MB)':>14}")
    print("-" * 68)
    for row in results:
        print(
            f"{row['name']:<24} {row['median_seconds']:>12.6f} "
            f"{row['stdev_seconds']:>12.6f} {row['peak_memory_mb']:>14.3f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="USEL CPU/memory benchmark suite.")
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results.json / results.csv",
    )
    parser.add_argument(
        "--repeats", type=int, default=REPEATS, help="Number of timed repeats per workload"
    )
    parser.add_argument(
        "--workload", type=str, default=None, help="Run only this workload (default: all)"
    )
    args = parser.parse_args(argv)

    env = environment()
    results = run_benchmark(name=args.workload, repeats=args.repeats)
    print_summary(results)
    write_outputs(results, env, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
