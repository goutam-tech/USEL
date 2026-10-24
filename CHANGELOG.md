# Changelog

All notable changes to USEL are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `usel.utils.benchmarking`: reproducible CPU-time and peak-memory benchmark helpers
  (`run_benchmark`, `BenchmarkResult`, `environment_info`, `format_table`, `write_json`).
- `benchmarks/run_benchmarks.py`: seeded benchmark suite (matrix multiplication, FFT, RK4)
  with `--repeats`, `--warmup`, `--quick`, `--no-memory` and `--json` options.
- `scripts/check_version.py`: verifies that `pyproject.toml`, `usel.__version__` and this
  changelog agree on the current version.
- `CHANGELOG.md` (the docs site already referenced it).

### Changed
- `pyproject.toml` version aligned with `usel.__version__` (`1.1.0`).

## [1.1.0]

### Added
- Core numerics: matrices, linear algebra, FFT transforms, root finding, ODE integration
  and quadrature.
- Schrödinger, Dirac and Maxwell equation modules.
- Black-Scholes pricing, Greeks, implied volatility and PDE/Monte Carlo solvers.
- Utilities: random generation, configuration loading, file I/O, logging, validation,
  profiling and the `usel` command line interface.

[Unreleased]: https://github.com/goutam-tech/usel/compare/1.1.0...HEAD
[1.1.0]: https://github.com/goutam-tech/usel/releases/tag/1.1.0
