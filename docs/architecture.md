# Architecture

usel is organized as a set of small, focused modules under
`src/usel/`, each responsible for a single concern:

```text
usel/
├── math/        # Matrix class, algebra, FFT transforms
├── linalg/       # Functional decomposition wrappers (SVD, QR, LU, Cholesky, eigen)
├── solvers/      # Root finding, ODE integration, numerical integration
├── random/       # Reproducible random generation
├── utils/        # Timing, profiling, validation, precision control
├── logging/      # Console/file loggers, performance logger
├── config/       # YAML/JSON/TOML configuration loading
├── io/           # CSV/JSON/NumPy binary/text file I/O
├── exceptions/   # Custom exception hierarchy
└── cli/          # Command line interface
```

## Design Principles

- **Composition over inheritance** — modules expose plain functions and a
  small number of simple classes (`Matrix`, `SolverResult`, `Timer`,
  `PerformanceLogger`) rather than deep class hierarchies.
- **Pure functions where possible** — solvers and transforms accept and
  return data without mutating shared state.
- **No global mutable state** — the one exception, global precision in
  `usel.utils.precision`, is explicit and scoped to formatting output.
- **Fail loudly, fail clearly** — invalid input raises a typed exception
  from `usel.exceptions` (`MatrixError`, `SolverError`, `ConfigError`,
  `ValidationError`, `IOError_`) rather than failing silently or returning
  `None`.

## Data Flow

Most numerical routines operate on either raw `numpy.ndarray` objects or the
`Matrix` wrapper, which exposes `.data` for direct NumPy/SciPy
interoperability. This keeps usel composable with the broader scientific
Python ecosystem while providing a clean, typed, object-oriented API on top.

## Future Compatibility

The module boundaries above are designed to remain stable as usel grows:

- **v2.0 — Quantum Mechanics Engine** will introduce a new top-level
  `quantum` package built on the existing `linalg` and `solvers` primitives.
- **v3.0 — Computational Physics** will add domain-specific solvers reusing
  the `solvers.ode` and `solvers.integration` interfaces.
- **v4.0 — Quantum Information** will add qubit/gate/circuit abstractions.
- **v5.0 — Research Toolkit** and **v6.0 — Plugin System** will build on top
  of the stable core without requiring breaking changes to `math`, `linalg`,
  or `solvers`.
