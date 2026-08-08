# USEL

USEL () is an open-source, CPU-only scientific computing framework focused on
quantum physics simulations and numerical methods.

Version 1.0 establishes the mathematical and computational foundation —
matrix and linear algebra utilities, numerical solvers, FFT-based frequency
analysis, random number generation, configuration/IO/logging utilities, and a
CLI — for future quantum mechanics, computational physics, and quantum
information modules.

## Features

- Runs efficiently on standard consumer hardware, no GPU required
- Operates within 8 GB RAM
- Cross-platform (Windows, Linux, macOS)
- Modular, extensible architecture
- Fully typed, PEP 8 compliant codebase

## Installation

```bash
uv sync
uv run usel version
```

Or, for local development:

```bash
git clone https://github.com/usel/usel.git
cd usel
uv sync --group dev
```

See the [examples](examples/) directory for more, including eigenvalue
solving, ODE integration, FFT analysis, and numerical integration.

## Command Line Interface

```bash
uv run usel version
uv run usel doctor
uv run usel benchmark --size 200
uv run usel info
uv run usel config path/to/config.yaml
uv run usel test
```

## Modules

| Module         | Description                                      |
| -------------- | ------------------------------------------------ |
| `usel.math`    | Matrix creation, algebra, norms, FFT transforms  |
| `usel.linalg`  | Eigenvalues/vectors, SVD, QR, LU, Cholesky       |
| `usel.solvers` | Root finding, ODE solvers, numerical integration |
| `usel.random`  | Random matrices, vectors, and distributions      |
| `usel.utils`   | Timing, profiling, validation, precision control |
| `usel.logging` | Console/file loggers, performance logger         |
| `usel.config`  | YAML/JSON/TOML configuration loading             |
| `usel.io`      | CSV/JSON/NumPy binary/text file I/O              |
| `usel.cli`     | Command line interface                           |

## Non-Goals (v1.0)

This release intentionally excludes the Schrödinger equation, quantum
circuits/gates/qubits, machine learning, chemistry simulation, GPU/cloud
computing, and AI integration. These are planned for future releases (see
[CHANGELOG.md](CHANGELOG.md) and the roadmap below).

## Roadmap

- v1.0 — Schrödinger Module
  - Non-relativistic quantum state evolution
  - Grid, operator, and Hamiltonian construction
  - Multiple time-evolution schemes (Crank-Nicolson, split-operator, RK4)

- v2.0 — Dirac Equation Module
  - Relativistic 1D Dirac equation solver
  - Spinor algebra and Pauli/Dirac matrix library
  - Matrix-exponential propagation

- v3.0 — Maxwell Equation Module
  - Electromagnetic field simulations
  - Symbolic and numerical field solvers

- v4.0 — Black-Scholes Module
  - Analytical option pricing, Greeks, and put-call parity
  - Implied volatility and Black-Scholes PDE solver
  - Monte Carlo pricing with variance reduction

Still more planning is going on this.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run black --check .
uv run pytest --cov=./ --cov-report xml
```

## Documentation

Full documentation is built with MkDocs:

```bash
uv run mkdocs serve
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines.

## Security

USEL performs no telemetry, no network communication, and no data
collection. It is fully offline capable.

## License

MIT — see [LICENSE](LICENSE).