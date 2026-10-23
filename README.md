# USEL — Unified Scientific Equation Library

**USEL** is an open-source, CPU-only scientific computing framework for quantum physics simulations and numerical methods, designed to make scientific computation accessible on standard hardware.

Version 1.1 establishes a comprehensive mathematical and computational foundation spanning:
- **Core numerics**: matrix/linear algebra, FFT transforms, numerical solvers (root finding, ODE integration, quadrature)
- **Quantum mechanics**: Schrödinger equation solvers (bound states, time evolution, wave packets, tunnelling)
- **Relativistic quantum mechanics**: Dirac equation solvers (spinors, propagators, Hamiltonians, potentials)
- **Computational electromagnetics**: Maxwell equation solvers (symbolic fields, potentials, materials, boundary conditions)
- **Quantitative finance**: Black-Scholes option pricing (analytical formulas, Greeks, implied volatility, PDE/Monte Carlo solvers)
- **Utilities**: reproducible random generation, configuration (YAML/JSON/TOML), file I/O, logging, validation, profiling, CLI

All modules run efficiently on standard consumer hardware (≤ 8 GB RAM, no GPU required) across Windows, Linux, and macOS.

---

## Quick Start

### Prerequisites
- Python **3.11** or **3.12**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

**Option 1: Using uv (recommended, fastest)**
```bash
git clone https://github.com/goutam-tech/usel.git
cd usel
uv sync --group dev
```

**Option 2: Using pip**
```bash
git clone https://github.com/goutam-tech/usel.git
cd usel
pip install -e ".[dev]"
```

**Option 3: Install from PyPI (when published)**
```bash
pip install usel
```

### Verify Installation
```bash
uv run usel doctor
# Expected: All checks passed. usel is ready to use.

uv run usel version
# Expected: usel version 1.1.0

uv run python -c "from usel.math import Matrix; print(Matrix.identity(3))"
# Expected: 3x3 identity matrix printed
```

---

## Your First Computation (30 seconds)

### Quick-Start: Solve the Quantum Harmonic Oscillator
```python
# quickstart_ho.py
import numpy as np
from usel.schrodinger import SchrodingerSolver, gaussian_wavepacket

# Define spatial grid and harmonic potential V(x) = 0.5 * x^2
x = np.linspace(-8, 8, 512)
V = 0.5 * x**2

solver = SchrodingerSolver(x, V)

# 1. Compute bound state eigenenergies (analytical: n + 0.5)
result = solver.eigenstates(n_states=6)
print("Harmonic Oscillator — Energy Eigenvalues")
print("  n   Computed     Analytical")
for i in range(6):
    analytical = i + 0.5
    print(f"  {i}   {result.energies[i]:10.6f}   {analytical:10.6f}")

# 2. Time-evolve a Gaussian wave packet
psi0 = gaussian_wavepacket(x, x0=-3.0, sigma=0.5, k0=5.0)
td = solver.solve_time_dependent(psi0, t_end=1.0, dt=0.02)

print(f"\nProbability conservation: initial={td.norm[0]:.6f}, final={td.norm[-1]:.6f}")
print(f"<x> initial: {td.expectation_x(0):.4f}, <x> final: {td.expectation_x(-1):.4f}")
```

Run it:
```bash
uv run python quickstart_ho.py
```

**Expected output:**
```
Harmonic Oscillator — Energy Eigenvalues
  n   Computed     Analytical
  0     0.500000     0.500000
  1     1.500000     1.500000
  2     2.500000     2.500000
  3     3.500000     3.500000
  4     4.500000     4.500000
  5     5.500000     5.500000

Probability conservation: initial=1.000000, final=1.000000
<x> initial: -3.0000, <x> final: 3.0000
```

---

## Implemented Features

| Module | Status | Key Capabilities |
|--------|--------|------------------|
| `usel.math` | ✅ Implemented | `Matrix` class (creation, algebra, norms, det, inverse, trace), FFT (`fft`, `ifft`, `frequency_analysis`) |
| `usel.linalg` | ✅ Implemented | Eigenvalues/vectors, SVD, QR, LU, Cholesky decompositions |
| `usel.solvers` | ✅ Implemented | Root finding (Newton, bisection, secant, fixed-point, gradient descent), ODE solvers (Euler, improved Euler, RK2, RK4), numerical integration (trapezoidal, Simpson, Gaussian quadrature) |
| `usel.random` | ✅ Implemented | Reproducible random matrices/vectors, Gaussian/uniform numbers, seeded state management |
| `usel.utils` | ✅ Implemented | `Timer`/`timeit`, profiling (`profile`, `memory_usage_mb`), validation (`validate_*`), precision control |
| `usel.config` | ✅ Implemented | Load/save YAML, JSON, TOML configuration files |
| `usel.io` | ✅ Implemented | CSV, JSON, NumPy binary (`.npy`), plain text read/write |
| `usel.logging` | ✅ Implemented | Console/file loggers, debug mode, `PerformanceLogger` |
| `usel.exceptions` | ✅ Implemented | Typed exception hierarchy (`uselError`, `MatrixError`, `SolverError`, `ConfigError`, `ValidationError`, `IOError_`, `PhysicsError`) |
| `usel.cli` | ✅ Implemented | `usel version`, `doctor`, `benchmark`, `info`, `config`, `test` |
| `usel.schrodinger` | ✅ Implemented | 1D time-independent/dependent Schrödinger solver, grids, Hamiltonians, operators, propagators (Crank-Nicolson, split-operator, RK4), wavefunctions, observables, normalization, boundary conditions, visualization |
| `usel.dirac` | ✅ Implemented | 1D Dirac equation solver, spinors, gamma/Pauli/Dirac matrices, Hamiltonians, propagators, potentials, boundary conditions, validation, visualization |
| `usel.maxwell` | ✅ Implemented | Symbolic Maxwell solvers (core fields, potentials, materials, boundary conditions, energy, integral forms, covariant formulation, waves, special cases) |
| `usel.black_scholes` | ✅ Implemented | Analytical pricing, Greeks (Δ, Γ, Θ, ν, ρ), put-call parity, implied volatility (Newton/bisection), finite-difference PDE solver, Monte Carlo with variance reduction, distributions |

---

## Planned Features (Not Yet Implemented)

The following are **planned for future releases** and are **not available in v1.1**:

| Module | Planned Version | Description |
|--------|----------------|-------------|
| Quantum circuits/gates/qubits | v2.0+ | Quantum information toolkit |
| Machine learning integration | v3.0+ | Neural networks, differentiable physics |
| GPU/cloud computing | v3.0+ | CUDA/ROCm backends, distributed computing |
| Chemistry simulation | v4.0+ | Molecular dynamics, electronic structure |
| 2D/3D Schrödinger/Maxwell | v4.0+ | Higher-dimensional PDE solvers |

> **Note:** The roadmap in earlier versions listed some of these under specific version numbers. The current roadmap is under revision. See [CHANGELOG.md](CHANGELOG.md) and [GitHub Issues](https://github.com/goutam-tech/usel/issues) for the latest planning.

---

## Comprehensive Examples

All examples are in the [`example/`](example/) directory and are verified to run.

### Core Numerics (`example/maths/`)
```bash
# Matrix operations
uv run python example/maths/example_matrix_multiplication.py

# Eigenvalue solver
uv run python example/maths/example_eigenvalue_solver.py

# FFT frequency analysis
uv run python example/maths/example_fft_analysis.py

# Numerical integration (trapezoidal, Simpson, Gaussian quadrature)
uv run python example/maths/example_numerical_integration.py

# ODE solver (RK4 exponential decay)
uv run python example/maths/example_ode_solver.py
```

### Schrödinger Equation (`example/schrodingers/`)
```bash
# Spatial grids
uv run python example/schrodingers/example_schrodinger_grids.py

# Operators & Hamiltonian construction
uv run python example/schrodingers/example_schrodinger_operators_hamiltonian.py

# Particle in a box (infinite square well)
uv run python example/schrodingers/example_schrodinger_particle_in_box.py

# Quantum harmonic oscillator (eigenstates + time evolution)
uv run python example/schrodingers/example_schrodinger_harmonic_oscillator.py

# Quantum tunnelling through a barrier
uv run python example/schrodingers/example_schrodinger_quantum_tunneling.py

# Normalization & observables
uv run python example/schrodingers/example_schrondinger_normalization_observables.py

# Time evolution, validation & visualization
uv run python example/schrodingers/example_schrondinger_evol_valid_visualization.py

# Solvers comparison
uv run python example/schrodingers/example_schrondinger_solvers.py
```

### Dirac Equation (`example/dirac/`)
```bash
# Foundation: spinors, gamma matrices
uv run python example/dirac/example_dirac_foundation.py

# Grids & potentials
uv run python example/dirac/example_dirac_grid_potential.py

# Free particle propagation
uv run python example/dirac/example_dirac_free_particle.py

# Operators & Hamiltonian
uv run python example/dirac/example_dirac_operator_hamiltonian.py

# Propagator validation
uv run python example/dirac/example_dirac_propagator_validation.py

# Visualization
uv run python example/dirac/example_dirac_visualization.py
```

### Maxwell Equations (`example/maxwell/`)
```bash
uv run python example/maxwell/example_1.py
uv run python example/maxwell/example_2.py
uv run python example/maxwell/example_3.py
```

### Black-Scholes (`example/black_scholes/`)
```bash
# Distributions & parameters
uv run python example/black_scholes/example1_dist_para.py

# Pricing & put-call parity
uv run python example/black_scholes/example_pricing_and_parity.py

# Greeks & implied volatility
uv run python example/black_scholes/example_greeks_and_implied_volatility.py

# Three-way validation (analytical, PDE, Monte Carlo)
uv run python example/black_scholes/example_three_way_validation.py
```

---

## Module API Overview

### `usel.math` — Matrices & Transforms
```python
from usel.math import Matrix, fft, ifft, frequency_analysis

# Matrix creation
Matrix([[1, 2], [3, 4]])  # from nested list
Matrix.identity(3)  # identity matrix
Matrix.random(4, 4, seed=42)  # reproducible random matrix
Matrix.zeros(3, 3)  # zero matrix

# Algebra
a @ b  # matrix multiplication
a.T  # transpose
a.determinant()  # determinant
a.trace()  # trace
a.inverse()  # inverse (raises MatrixError if singular)
a.norm()  # Frobenius norm

# FFT
fft(signal)  # forward FFT
ifft(spectrum)  # inverse FFT
frequency_analysis(signal, sr)  # returns (frequencies, magnitudes)
```

### `usel.linalg` — Decompositions
```python
from usel.linalg import eigenvalues, eigenvectors, svd, qr, lu, cholesky
from usel.math import Matrix

m = Matrix([[4, 1], [2, 3]])

values = eigenvalues(m)  # eigenvalues only
values, vectors = eigenvectors(m)  # eigenvalues + eigenvectors
U, S, Vt = svd(m)  # singular value decomposition
Q, R = qr(m)  # QR decomposition
L, U = lu(m)  # LU decomposition
L = cholesky(m)  # Cholesky (requires SPD matrix)
```

### `usel.solvers` — Root Finding, ODEs, Integration
```python
from usel.solvers import (
    newton,
    bisection,
    secant,
    fixed_point,
    gradient_descent,
    euler,
    improved_euler,
    rk2,
    rk4,
    trapezoidal,
    simpson,
    gaussian_quadrature,
)

# Root finding
root = newton(f, df, x0=1.0, tol=1e-10, max_iter=100)
root = bisection(f, a=0, b=2, tol=1e-10)

# ODE: dy/dt = f(t, y)
t, y = rk4(f, t0=0.0, y0=1.0, t_end=10.0, h=0.01)

# Integration: ∫ f(x) dx from a to b
val = trapezoidal(f, a, b, n=1000)
val = simpson(f, a, b, n=1000)
val = gaussian_quadrature(f, a, b, n=10)
```

### `usel.random` — Reproducible Random Generation
```python
from usel.random import (
    random_matrix,
    random_vector,
    gaussian_numbers,
    uniform_numbers,
    random_state,
)

m = random_matrix(3, 3, seed=42)  # 3x3 random matrix
v = random_vector(5, seed=42)  # length-5 random vector
nums = gaussian_numbers(100, mean=0, std=1, seed=42)
nums = uniform_numbers(100, low=0, high=1, seed=42)
state = random_state(seed=42)  # get/set NumPy random state
```

### `usel.utils` — Timing, Profiling, Validation, Precision
```python
from usel.utils import Timer, timeit, profile, memory_usage_mb
from usel.utils import validate_positive, validate_range, validate_shape, validate_type
from usel.utils import set_global_precision, round_to_precision

# Timing
with Timer() as t:
    result = expensive_computation()
print(f"Took {t.elapsed:.4f} seconds")


@timeit
def my_function(): ...


# Profiling
@profile
def my_function(): ...


mem_mb = memory_usage_mb()

# Validation (raise ValidationError on failure)
validate_positive(x, "x")
validate_range(x, 0, 10, "x")
validate_shape(arr, (3, 3), "arr")
validate_type(x, int, "x")

# Output precision
set_global_precision(6)
print(round_to_precision(3.1415926535))  # 3.141593
```

### `usel.config` — Configuration Files
```python
from usel.config import load_config, save_config

# Load from .yaml, .json, or .toml
config = load_config("config.yaml")

# Save to any supported format
save_config(config, "output.toml")
```

### `usel.io` — File I/O
```python
from usel.io import (
    read_csv,
    write_csv,
    read_json,
    write_json,
    read_npy,
    write_npy,
    read_text,
    write_text,
)

# CSV
df = read_csv("data.csv")
write_csv(data, "output.csv")

# JSON
data = read_json("data.json")
write_json(data, "output.json")

# NumPy binary
arr = read_npy("data.npy")
write_npy(arr, "output.npy")

# Plain text
text = read_text("data.txt")
write_text("hello", "output.txt")
```

### `usel.logging` — Logging Utilities
```python
from usel.logging import get_console_logger, get_file_logger, set_debug_mode, PerformanceLogger

logger = get_console_logger("my_module")
logger.info("Message")

file_logger = get_file_logger("my_module", "app.log")

set_debug_mode(True)

perf = PerformanceLogger("my_module")
perf.start("operation")
# ... work ...
perf.stop("operation")
```

### `usel.exceptions` — Typed Exceptions
```python
from usel.exceptions import (
    uselError,
    MatrixError,
    SolverError,
    ConfigError,
    ValidationError,
    IOError_,
    PhysicsError,
)

try:
    result = solver.solve(...)
except SolverError as e:
    print(f"Solver failed: {e}")
except ValidationError as e:
    print(f"Invalid input: {e}")
```

### `usel.cli` — Command Line Interface
```bash
usel version          # Print version
usel info             # Platform & package info
usel doctor           # Environment health check
usel benchmark --size 200  # Matrix multiplication benchmark
usel config path.yaml # Load & validate config file
usel test             # Run test suite via pytest
usel test tests/src/solvers  # Run specific test path
```

### `usel.schrodinger` — 1D Schrödinger Equation
```python
from usel.schrodinger import (
    SchrodingerSolver,
    gaussian_wavepacket,
    build_hamiltonian,
    solve_energy_levels,
)
from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.operators import kinetic_energy_operator, momentum_operator, position_operator

# High-level solver
grid = SpatialGrid(xmin=-5, xmax=5, points=1000)
V = 0.5 * grid.x**2
solver = SchrodingerSolver(grid.x, V)

# Bound states
result = solver.eigenstates(n_states=5)
result.energies  # array of eigenvalues
result.eigenstates  # eigenvectors as columns

# Time-dependent evolution
psi0 = gaussian_wavepacket(grid.x, x0=-2, sigma=0.5, k0=3)
td_result = solver.solve_time_dependent(psi0, t_end=5.0, dt=0.01)
td_result.psi  # wavefunction at each time step
td_result.norm  # norm at each time step (conservation check)
td_result.expectation_x(i)  # <x> at time index i
td_result.expectation_p(i)  # <p> at time index i

# Low-level components
grid = SpatialGrid(xmin=0, xmax=1, points=500)
T = kinetic_energy_operator(grid)
x_op = position_operator(grid)
p_op = momentum_operator(grid)
H = build_hamiltonian(grid, V)
energies, states = solve_energy_levels(H, levels=5)
```

### `usel.dirac` — 1D Dirac Equation
```python
from usel.dirac import DiracSolver, DiracGrid
from usel.dirac.spinors import positive_energy_spinor, spinor_norm, spin_expectation
from usel.dirac.gamma_matrices import gamma_matrices_dirac
from usel.dirac.hamiltonian import build_dirac_hamiltonian

# High-level solver
grid = DiracGrid(xmin=-10, xmax=10, points=500)
x = grid.x
V = 0.5 * x**2  # harmonic potential
solver = DiracSolver(x, V, mass=1.0, c=1.0, hbar=1.0)

# Time evolution (4-component spinor)
import numpy as np

psi0 = np.zeros((4, len(x)), dtype=complex)
# Create a Gaussian wave packet in the upper components
psi0[0, :] = np.exp(-((x + 3) ** 2) / (2 * 0.5**2)) * np.exp(1j * 2.0 * x)
psi0[1, :] = np.exp(-((x + 3) ** 2) / (2 * 0.5**2)) * np.exp(1j * 2.0 * x) * 0.1
psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * grid.dx)

td_result = solver.solve(psi0, t_end=2.0, dt=0.005)
td_result.spinor  # shape (steps+1, 4, N)
td_result.probability  # probability density at each step
td_result.t  # time points
td_result.x  # spatial grid

# Low-level
g0, g1, g2, g3 = gamma_matrices_dirac()
H = build_dirac_hamiltonian(x, V, mass=1.0, c=1.0, hbar=1.0)
```

### `usel.maxwell` — Symbolic Electromagnetics
```python
from usel.maxwell import core, potentials, materials, boundary, energy
from usel.maxwell.constants import EPSILON_0, MU_0, C_LIGHT

# Symbolic fields (SymPy-based)
from usel.maxwell.core import N

x, y, z = N.x, N.y, N.z

# Point charge field
E = (q / (4 * sp.pi * EPSILON_0)) * (x * N.i + y * N.j + z * N.k) / (x**2 + y**2 + z**2) ** (3 / 2)
residual, ok = core.gauss_law_electric(E, rho=0, epsilon0=EPSILON_0)

# Potentials
A = (-B0 * y / 2) * N.i + (B0 * x / 2) * N.j
B = potentials.B_from_potentials(A)

# Materials
D = materials.D_from_E(E, materials.WATER)
E_recovered, ok = materials.round_trip_D_to_E(E, materials.WATER)

# Boundary conditions
R, T = boundary.reflection_transmission_coefficients(eps1, mu1, eps2, mu2, theta_i)
```

### `usel.black_scholes` — Option Pricing
```python
from usel.black_scholes import (
    call_price,
    put_price,
    delta_call,
    delta_put,
    gamma,
    vega,
    theta_call,
    theta_put,
    rho_call,
    rho_put,
    implied_volatility_call,
    implied_volatility_put,
    verify_put_call_parity,
    put_from_call,
    call_from_put,
    finite_difference_call_price,
    finite_difference_put_price,
    monte_carlo_call_price,
    monte_carlo_put_price,
)

# Analytical pricing (positional args: stock, strike, rate, vol, time, div_yield=0)
call = call_price(100, 100, 0.05, 0.2, 1.0)
put = put_price(100, 100, 0.05, 0.2, 1.0)

# Greeks (same positional signature)
Δ_call = delta_call(100, 100, 0.05, 0.2, 1.0)
Δ_put = delta_put(100, 100, 0.05, 0.2, 1.0)
Γ = gamma(100, 100, 0.05, 0.2, 1.0)
ν = vega(100, 100, 0.05, 0.2, 1.0)
Θ_call = theta_call(100, 100, 0.05, 0.2, 1.0)
Θ_put = theta_put(100, 100, 0.05, 0.2, 1.0)
ρ_call = rho_call(100, 100, 0.05, 0.2, 1.0)
ρ_put = rho_put(100, 100, 0.05, 0.2, 1.0)

# Implied volatility
iv = implied_volatility_call(10.45, 100, 100, 0.05, 1.0)

# Put-call parity
residual, ok = verify_put_call_parity(call, put, 100, 100, 0.05, 1.0)

# Alternative solvers
call_fd = finite_difference_call_price(100, 100, 0.05, 0.2, 1.0)
call_mc = monte_carlo_call_price(100, 100, 0.05, 0.2, 1.0, n_paths=10000)
```

---

## Supported Python Versions & Dependencies

| Requirement | Version |
|-------------|---------|
| Python | **≥ 3.11** (tested on 3.11, 3.12) |
| NumPy | ≥ 1.26 |
| SciPy | ≥ 1.11 |
| SymPy | ≥ 1.12 |
| PyYAML | ≥ 6.0 |
| TOML | ≥ 0.10.2 |
| Matplotlib | ≥ 3.8 (for visualization examples) |
| pytest | ≥ 9.1.1 (dev) |

**Dev dependencies:** `ruff`, `black`, `mypy`, `pytest-cov`

---

## Hardware Requirements & Known Limitations

| Requirement | Specification |
|-------------|---------------|
| **RAM** | ≤ 8 GB for typical workloads; large grids (10k+ points) may require more |
| **CPU** | Any x86_64 or ARM64; SIMD acceleration via NumPy/SciPy |
| **GPU** | **Not required**; no CUDA/ROCm support in v1.x |
| **Disk** | ~50 MB for package + dependencies |
| **OS** | Windows 10+, Linux (glibc ≥ 2.28), macOS 11+ |

**Known Limitations:**
- **1D only**: Schrödinger, Dirac, and Maxwell solvers operate in 1D spatial dimension
- **No GPU acceleration**: All computation runs on CPU via NumPy/SciPy
- **SymPy performance**: Maxwell symbolic operations can be slow for complex expressions
- **No distributed computing**: Single-process only; no MPI or multi-node support
- **Black-Scholes Monte Carlo**: Single-threaded; variance reduction helps but large samples take time

---

## Documentation

| Resource | Link |
|----------|------|
| **API Reference** | https://goutam-tech.github.io/usel/api/ |
| **Architecture Overview** | [docs/architecture.md](docs/architecture.md) |
| **Developer Guide** | [docs/developer_guide.md](docs/developer_guide.md) |
| **Contributing Guidelines** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |
| **Issue Tracker** | https://github.com/goutam-tech/usel/issues |
| **Discussions** | https://github.com/goutam-tech/usel/discussions |

**Serve docs locally:**
```bash
uv run mkdocs serve
# Open http://127.0.0.1:8000
```

---

## Development

### Setup Development Environment
```bash
git clone https://github.com/goutam-tech/usel.git
cd usel
./scripts/setup_dev_env.sh
# Or manually:
uv sync --group dev
uv run pre-commit install
```

### Quality Gates
```bash
# Run all checks (lint, format, type-check, tests)
./scripts/run_release_checks.sh

# Individual commands
uv run ruff check . --fix
uv run black .
uv run mypy src
uv run pytest --cov=usel --cov-report=term-missing
```

### Project Structure
```
usel/
├── src/usel/           # Package source
│   ├── math/           # Matrix, FFT
│   ├── linalg/         # Decompositions
│   ├── solvers/        # Root finding, ODE, integration
│   ├── random/         # Random generation
│   ├── utils/          # Timing, profiling, validation, precision
│   ├── config/         # YAML/JSON/TOML config
│   ├── io/             # File I/O
│   ├── logging/        # Loggers
│   ├── exceptions/     # Exception hierarchy
│   ├── cli/            # Command line interface
│   ├── schrodinger/    # 1D Schrödinger equation
│   ├── dirac/          # 1D Dirac equation
│   ├── maxwell/        # Symbolic Maxwell equations
│   └── black_scholes/  # Option pricing
├── tests/              # Unit tests (mirrors src/)
├── example/            # Runnable examples
├── docs/               # MkDocs documentation
├── scripts/            # Dev scripts
├── benchmarks/         # Performance benchmarks
└── frontend/           # Web demo (HTML/JS/CSS)
```

---

## Testing

```bash
# Full test suite
uv run pytest

# With coverage
uv run pytest --cov=usel --cov-report=term-missing

# Specific module
uv run pytest tests/src/solvers/
uv run pytest tests/src/schrodinger/

# Verbose output
uv run pytest -v
```

The suite includes a numerical-correctness baseline that validates core scientific
methods (linear algebra, FFT, quadrature, decompositions) against analytical
solutions with justified tolerances, plus boundary/invalid-input cases. These tests
are deterministic, run on CPU only, and require no network access.

## Benchmarking

Reproducible CPU and memory benchmarks live in ``benchmarks/`` and cover
implemented functionality only (matrix multiplication, FFT, ODE integration,
quadrature, root finding, Black-Scholes pricing, SVD). They record runtime and
peak memory across repeated runs, report median and spread, and export JSON/CSV
with environment metadata (Python, platform, CPU, dependency versions).

```bash
# Run every benchmark and write results.json / results.csv
uv run python benchmarks/bench_suite.py --out benchmark_results

# Run a single workload
uv run python benchmarks/bench_suite.py --workload fft --repeats 10

# Re-run to confirm the report tolerates normal timing variation
uv run python benchmarks/bench_suite.py --out benchmark_results --repeats 5
```

Benchmarks are kept separate from the correctness test suite and use no strict
timing thresholds, so they never block CI. Numbers depend on hardware, software
versions, and system load — no performance claim is made unless directly
supported by a recorded, reproducible measurement.

## Solver Diagnostics

USEL exposes convergence and termination diagnostics for its root-finding
solvers through the structured :class:`usel.solvers.root_finding.SolverResult`
return type. ODE and integration solvers are fixed-step / single-evaluation
methods with no convergence criterion, so their diagnostics are limited to
step count and completion status.

### Root-finding solvers

``newton``, ``bisection``, ``secant``, ``gradient_descent``, and
``fixed_point`` all return :class:`SolverResult` with these fields:

* ``root`` — approximate root or minimizer (float).
* ``iterations`` — number of iterations performed before termination (int).
* ``converged`` — ``True`` when the tolerance was met before ``max_iter``,
  ``False`` when the iteration limit was reached (bool).
* ``history`` — sequence of iterates starting with the initial guess
  (list[float]).
* ``residual`` — absolute value of the defining equation at ``root``
  (``|f(root)|`` for root finding, ``|df(root)|`` for gradient descent,
  ``|g(root) - root|`` for fixed-point iteration). Finite and meaningful
  whenever the algorithm defines it (float or ``None``).

Termination status is distinguished as follows:

* Successful convergence — ``converged=True``.
* Iteration limit reached — ``converged=False`` with ``iterations == max_iter``.
* Invalid input — raises :class:`usel.exceptions.SolverError` (e.g. same-sign
  bracket for bisection).
* Numerical breakdown — raises :class:`usel.exceptions.SolverError` (e.g. zero
  derivative for Newton, zero denominator for secant).

No convergence guarantees or error bounds are claimed beyond what the
underlying algorithm establishes.

### ODE solvers

``euler``, ``improved_euler``, ``rk2``, and ``rk4`` are fixed-step methods
that always run every step from ``t0`` to ``t_end``. They return ``(t, y)``
tuples; the step count is ``len(t) - 1`` and termination status is always
completed. No convergence criterion or residual is defined by these algorithms.

### Integration solvers

``trapezoidal``, ``simpson``, and ``gaussian_quadrature`` perform a single
deterministic evaluation and return a float. The subinterval count is ``n``
(``simpson`` promotes odd ``n`` to the next even number). No convergence
criterion or residual is defined by these algorithms.

Backward compatibility: existing solver return types and default behavior are
unchanged. The ``residual`` field on :class:`SolverResult` defaults to
``None`` for code that constructs it directly.

---

## Dependency & Security Checks

USEL audits its declared dependencies for known vulnerabilities using
``pip-audit`` (installed as a development dependency). The advisory database
is fetched from the Python Packaging Advisory Database, so the check requires
network access and its results are only as current as the database.

Run locally:

```bash
uv sync --group dev
uv run pip-audit --desc on
```

CI runs the same check on every push and pull request. A clean scan exits
with code 0; actionable vulnerability findings exit with code 1 and block
the run. Tool or network failures (unable to reach the advisory database)
are treated as scan errors, not as clean results — they are visible and
actionable in CI logs.

No secrets or credentials are scanned or exposed. Dependency updates are
justified and validated rather than applied as blind bulk upgrades.

---


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow
- Coding standards (PEP 8, type hints, NumPy docstrings)
- Pull request guidelines
- Issue reporting template
- Code of Conduct

---

## Security

USEL performs **no telemetry, no network communication, and no data collection**. It is fully offline-capable by design.

To report a security vulnerability, please email the maintainers directly (see [pyproject.toml](pyproject.toml) for contact info) rather than opening a public issue.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use USEL in academic work, please cite:

```bibtex
@software{usel2025,
  title = {USEL: Unified Scientific Equation Library},
  author = {Gotur, Goutam Parashuram and Raaghavindhra, K R},
  version = {1.1.0},
  year = {2025},
  url = {https://github.com/goutam-tech/usel},
  license = {MIT}
}
```

---

## Acknowledgments

Built on the shoulders of giants: **NumPy**, **SciPy**, **SymPy**, **Matplotlib**, **PyYAML**, and the broader scientific Python ecosystem.