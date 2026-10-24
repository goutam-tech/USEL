"""Benchmark dense and sparse Crank–Nicolson propagation.

Run from the repository root:

    python benchmarks/benchmark_sparse_schrodinger.py
    python benchmarks/benchmark_sparse_schrodinger.py --repeats 5 --sizes 200 400 800

The benchmark compares runtime, traced peak memory, and estimated matrix
storage. It also checks that dense and sparse propagation agree numerically
and that both methods preserve the discrete wavefunction norm.

Results are written to benchmark-results/sparse-methods.json by default.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


def initial_problem(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a normalized initial state and harmonic-oscillator potential."""
    x = np.linspace(-8.0, 8.0, n)
    dx = float(x[1] - x[0])
    potential = 0.5 * x**2

    psi = np.exp(-(x**2) / 2.0).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi /= norm

    return x, potential, psi, dx


def run_dense(
    n: int,
    steps: int,
    dt: float,
) -> tuple[np.ndarray, int]:
    """Run dense Crank–Nicolson propagation and return state and storage."""
    _, potential, psi, dx = initial_problem(n)

    hbar = 1.0
    mass = 1.0
    kinetic_coeff = hbar**2 / (2.0 * mass * dx**2)

    diagonal = 2.0 * kinetic_coeff + potential
    off_diagonal = np.full(n - 1, -kinetic_coeff)

    hamiltonian = np.diag(diagonal)
    hamiltonian += np.diag(off_diagonal, k=1)
    hamiltonian += np.diag(off_diagonal, k=-1)

    identity = np.eye(n, dtype=np.complex128)
    factor = 0.5j * dt / hbar

    matrix_a = identity + factor * hamiltonian
    matrix_b = identity - factor * hamiltonian

    lu, pivots = lu_factor(matrix_a)

    history = np.empty((steps + 1, n), dtype=np.complex128)
    history[0] = psi

    for step in range(steps):
        rhs = matrix_b @ history[step]
        history[step + 1] = lu_solve((lu, pivots), rhs)

    # Approximate payload of the principal matrices and dense LU factors.
    matrix_payload = sum(
        array.nbytes
        for array in (
            hamiltonian,
            matrix_a,
            matrix_b,
            lu,
        )
    )

    return history[-1].copy(), int(matrix_payload)


def run_sparse(
    n: int,
    steps: int,
    dt: float,
) -> tuple[np.ndarray, int]:
    """Run sparse Crank–Nicolson propagation and return state and storage."""
    _, potential, psi, dx = initial_problem(n)

    hbar = 1.0
    mass = 1.0
    kinetic_coeff = hbar**2 / (2.0 * mass * dx**2)

    diagonal = 2.0 * kinetic_coeff + potential
    off_diagonal = np.full(n - 1, -kinetic_coeff)

    hamiltonian = diags(
        diagonals=[off_diagonal, diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csc",
        dtype=np.float64,
    )

    identity = eye(n, format="csc", dtype=np.complex128)
    factor = 0.5j * dt / hbar

    matrix_a = (identity + factor * hamiltonian).tocsc()
    matrix_b = (identity - factor * hamiltonian).tocsr()

    lu = splu(matrix_a)

    history = np.empty((steps + 1, n), dtype=np.complex128)
    history[0] = psi

    for step in range(steps):
        rhs = matrix_b @ history[step]
        history[step + 1] = lu.solve(rhs)

    # Estimate storage for sparse matrices and sparse LU factors.
    matrix_payload = sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        for matrix in (hamiltonian, matrix_a, matrix_b, lu.L, lu.U)
    )

    return history[-1].copy(), int(matrix_payload)


def measure(
    method: Callable[[int, int, float], tuple[np.ndarray, int]],
    n: int,
    steps: int,
    dt: float,
    repeats: int,
) -> dict[str, Any]:
    """Measure runtime and traced peak memory over repeated executions."""
    runtimes: list[float] = []
    peak_memories: list[int] = []
    matrix_payloads: list[int] = []
    final_state: np.ndarray | None = None

    for _ in range(repeats):
        tracemalloc.start()
        try:
            start = time.perf_counter()
            state, matrix_payload = method(n, steps, dt)
            elapsed = time.perf_counter() - start
            _, peak_memory = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        runtimes.append(elapsed)
        peak_memories.append(peak_memory)
        matrix_payloads.append(matrix_payload)
        final_state = state

    assert final_state is not None

    return {
        "median_runtime_seconds": statistics.median(runtimes),
        "min_runtime_seconds": min(runtimes),
        "runtime_stdev_seconds": (
            statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
        ),
        "median_traced_peak_bytes": int(statistics.median(peak_memories)),
        "median_matrix_payload_bytes": int(
            statistics.median(matrix_payloads)
        ),
        "repeats": repeats,
        "_final_state": final_state,
    }


def main() -> None:
    """Parse options, run benchmarks, validate results, and save JSON."""
    parser = argparse.ArgumentParser(
        description="Compare dense and sparse Schrödinger propagation."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per method and grid size (default: 3).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[200, 400, 800],
        help="Grid sizes to benchmark (default: 200 400 800).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of propagation steps (default: 20).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Time step (default: 0.005).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/sparse-methods.json"),
        help="Output JSON path.",
    )

    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.steps < 1
        or not np.isfinite(args.dt)
        or args.dt <= 0
        or not args.sizes
        or any(n < 3 for n in args.sizes)
    ):
        parser.error(
            "repeats and steps must be positive, dt must be finite and "
            "positive, and sizes must be >= 3"
        )

    results: list[dict[str, Any]] = []

    for n in args.sizes:
        print(f"Benchmarking grid size n={n} ...", flush=True)

        dense = measure(
            run_dense,
            n,
            args.steps,
            args.dt,
            args.repeats,
        )
        sparse = measure(
            run_sparse,
            n,
            args.steps,
            args.dt,
            args.repeats,
        )

        dense_state = dense.pop("_final_state")
        sparse_state = sparse.pop("_final_state")

        dx = 16.0 / (n - 1)

        max_abs_difference = float(
            np.max(np.abs(dense_state - sparse_state))
        )
        final_states_allclose = bool(
            np.allclose(
                dense_state,
                sparse_state,
                rtol=1e-8,
                atol=1e-10,
            )
        )

        dense_norm = float(
            np.sqrt(np.sum(np.abs(dense_state) ** 2) * dx)
        )
        sparse_norm = float(
            np.sqrt(np.sum(np.abs(sparse_state) ** 2) * dx)
        )

        norms_conserved = bool(
            np.isclose(dense_norm, 1.0, rtol=1e-8, atol=1e-10)
            and np.isclose(sparse_norm, 1.0, rtol=1e-8, atol=1e-10)
        )

        dense_runtime = dense["median_runtime_seconds"]
        sparse_runtime = sparse["median_runtime_seconds"]

        speedup = (
            float(dense_runtime / sparse_runtime)
            if sparse_runtime > 0
            else None
        )

        results.append(
            {
                "grid_size": n,
                "steps": args.steps,
                "dt": args.dt,
                "dense": dense,
                "sparse": sparse,
                "speedup_dense_over_sparse": speedup,
                "correctness": {
                    "final_states_allclose": final_states_allclose,
                    "max_absolute_state_difference": max_abs_difference,
                    "dense_final_norm": dense_norm,
                    "sparse_final_norm": sparse_norm,
                    "norms_conserved": norms_conserved,
                },
            }
        )

        print(
            f"  dense: {dense_runtime:.6f}s | "
            f"sparse: {sparse_runtime:.6f}s | "
            f"speedup: {speedup:.2f}x"
            if speedup is not None
            else f"  dense: {dense_runtime:.6f}s | sparse: {sparse_runtime:.6f}s",
            flush=True,
        )
        print(
            f"  states agree: {final_states_allclose} | "
            f"norms conserved: {norms_conserved} | "
            f"max difference: {max_abs_difference:.3e}",
            flush=True,
        )

    output = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "methodology": {
            "integrator": "Crank-Nicolson",
            "dense_solver": "scipy.linalg.lu_factor / lu_solve",
            "sparse_solver": "scipy.sparse.linalg.splu",
            "steps": args.steps,
            "dt": args.dt,
            "repeats": args.repeats,
            "grid_sizes": args.sizes,
            "timing_statistic": "median",
            "memory_measurement": (
                "tracemalloc peak; Python-traced allocations only"
            ),
            "matrix_storage_estimate": (
                "Principal matrix and factor payloads; excludes Python "
                "object overhead and temporary allocations"
            ),
            "normalization": (
                "Discrete physical norm: sqrt(sum(abs(psi)**2) * dx)"
            ),
        },
        "results": results,
    }

    if not all(
        item["correctness"]["final_states_allclose"]
        and item["correctness"]["norms_conserved"]
        for item in results
    ):
        raise SystemExit(
            "Dense/sparse agreement or norm-conservation validation failed"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
