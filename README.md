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

## Quick Start

```python
from usel.math import Matrix

A = Matrix.random(100, 100)
B = Matrix.identity(100)
C = A @ B
print(C.shape)
```

```python
from usel.solvers import rk4

def decay(t, y):
    return -y

t, y = rk4(decay, t0=0.0, y0=1.0, t_end=5.0, h=0.01)
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

- v1.0 — Scientific Computing Foundation
  - Core numerical engine
  - Linear algebra utilities
  - Matrix operations
  - Numerical solvers
  - FFT and mathematical utilities
  - Configuration, logging, and CLI support

- v2.0 — Quantum Physics Engine
  - Schrödinger equation module
  - Quantum wavefunction utilities
  - Potential models
  - Hamiltonian construction
  - Time evolution and visualization

- v3.0 — Relativistic & Computational Physics
  - Dirac equation module
  - Spinor algebra
  - Gamma matrix operations
  - Relativistic particle simulations
  - Maxwell equation module
  - Electromagnetic field simulations

Still more planning is going on this.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy src
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

usel performs no telemetry, no network communication, and no data
collection. It is fully offline capable.

## License

MIT — see [LICENSE](LICENSE).