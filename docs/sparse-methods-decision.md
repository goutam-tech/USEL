# Sparse Numerical Methods Assessment (SCRUM-12)

## Status

Audit, benchmarking, sparse implementation, and regression testing are complete. Benchmark correctness and norm-conservation checks passed. Final review and commit are pending.

## Scope and Audit

- `SchrodingerSolver.eigenstates()` already constructs the one-dimensional finite-difference Hamiltonian as a SciPy CSR matrix and uses `scipy.sparse.linalg.eigsh` to calculate a limited number of eigenstates.
- The finite-difference Hamiltonian is tridiagonal, with at most three non-zero entries per row.
- `SchrodingerSolver.solve_time_dependent()` originally constructed dense `H`, `A`, and `B` arrays and used dense `scipy.linalg.lu_factor` and `lu_solve` for Crank–Nicolson propagation.
- `pyproject.toml` already declares NumPy and SciPy. No additional dependency is required for the sparse implementation.
- The public `hamiltonian` property intentionally returns a dense NumPy array. Changing its public return type is outside this assessment's scope.

## Candidate Workload and Comparison

`benchmarks/benchmark_sparse_schrodinger.py` compares dense and sparse Crank–Nicolson propagation using equivalent one-dimensional harmonic-oscillator finite-difference operators.

The dense implementation uses NumPy arrays and SciPy's dense LU factorization. The sparse implementation uses SciPy sparse matrices and `scipy.sparse.linalg.splu`.

Both implementations construct the Hamiltonian and propagation matrices, factorize the Crank–Nicolson system, execute the same number of time steps, and compare their final wavefunctions.

The benchmark measures repeated execution times, traced peak memory, and estimated matrix and factor storage. It also checks final-state agreement and physical discrete norm conservation.

Run the benchmark from the repository root:

```bash
python benchmarks/benchmark_sparse_schrodinger.py --repeats 5 --sizes 200 400 800 --steps 20 --dt 0.005 --output benchmark-results/sparse-methods.json
```

The JSON output records Python and platform details, NumPy and SciPy versions, runtime summaries, numerical differences, norm measurements, traced memory, and matrix-storage estimates.

`tracemalloc` does not provide a complete process-resident memory measurement for native NumPy and SciPy allocations. The memory figures must therefore be interpreted as traced allocations and estimated matrix payload, not total process memory or guaranteed peak resident memory.

The benchmark has no timing threshold. It fails if dense and sparse final states disagree beyond `rtol=1e-8` and `atol=1e-10`, or if either method fails the configured norm-conservation check.

## Decision

**Decision: Proceed with the sparse Crank–Nicolson implementation.**

The benchmark compared dense and sparse implementations of the one-dimensional harmonic oscillator using 200, 400, and 800 grid points, 20 time steps, and five repetitions per configuration.

The latest local measurements were:

| Grid size | Dense median runtime | Sparse median runtime | Speedup |
|---:|---:|---:|---:|
| 200 | 0.106079 s | 0.002147 s | 49.41× |
| 400 | 0.173906 s | 0.002587 s | 67.22× |
| 800 | 0.314044 s | 0.003013 s | 104.24× |

The sparse implementation was faster for all three tested grid sizes. The measured speedup increased with grid size in this run. These results support using sparse matrices and sparse factorization for the tested workload; they are not universal performance guarantees.

### Numerical Validation

All three grid sizes passed the benchmark's final-state agreement and norm-conservation checks.

| Grid size | Final states agree within tolerance | Norms conserved | Maximum absolute state difference |
|---:|:---:|:---:|---:|
| 200 | Yes | Yes | 8.981 × 10⁻¹⁶ |
| 400 | Yes | Yes | 1.323 × 10⁻¹⁵ |
| 800 | Yes | Yes | 3.732 × 10⁻¹⁵ |

The final states agreed within `rtol=1e-8` and `atol=1e-10`. Both implementations preserved the physical discrete norm within the configured tolerance.

These measurements provide evidence that the sparse implementation maintains numerical behavior comparable to the dense reference for the tested problem while substantially reducing runtime.

### Implementation and Regression Testing

The time-dependent solver now uses sparse matrices for the internal Crank–Nicolson calculations and sparse LU factorization through `splu`.

The factorization is performed once per propagation call and reused across time steps. The existing public dense `hamiltonian` property, result-history behavior, and `TimeDependentResult` interface remain unchanged.

A regression test was added to check the propagated wavefunction shape, probability history, time points, and norm conservation.

Validation completed so far:

- Focused Schrödinger solver tests: **25 passed**.
- Full test suite: **1,107 passed**.
- Benchmark: all three grid sizes passed final-state agreement and norm-conservation checks.

The full test suite reported 12 warnings, with no test failures.

**Conclusion:** The audit, benchmark results, and regression tests support retaining the sparse Crank–Nicolson implementation. Performance thresholds should remain outside ordinary CI because timings depend on the execution environment.

## Limitations

- The benchmark uses a representative one-dimensional harmonic oscillator, not every potential or workload supported by USEL.
- The reported timings come from one Windows environment and are not universal performance guarantees.
- Dense and sparse numerical results need not be bit-for-bit identical; the benchmark compares them using explicit tolerances.
- Norm conservation is verified for the tested configurations, not every possible input or simulation.
- `tracemalloc` measurements and estimated matrix payload do not represent complete process memory usage.
- The benchmark does not impose a wall-clock performance threshold in CI.