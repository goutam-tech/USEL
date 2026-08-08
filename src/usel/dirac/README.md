# Dirac Equation Module

The Dirac module is a relativistic quantum mechanics component of **USEL
(Universal Scientific Engine Library)**.

It provides numerical solvers for the 1D Dirac equation, with spinor
algebra and Pauli/Dirac matrix definitions. The module is designed for
CPU-based scientific computing and works efficiently on standard consumer
hardware without GPU requirements.

---

## Overview

The module provides reusable components for relativistic quantum
simulations:

- Free-particle plane-wave spinor evolution
- Spatial Dirac wave-packet propagation
- Complete Pauli and Dirac matrix library
- Spinor normalisation and expectation values

The architecture is designed to support future extensions in relativistic
and computational physics research.

---

## Mathematical Foundation

The module is based on the time-dependent Dirac equation:

\[
i\hbar\frac{\partial\psi}{\partial t}=H_D\psi
\]

where the Dirac Hamiltonian is:

\[
H_D = c\,\boldsymbol{\alpha}\cdot\mathbf{p} + \beta m c^2 + V(x)
\]

- `ψ` represents the two-component spinor wavefunction
- `α`, `β` represent the Dirac matrices
- `c` represents the speed of light
- `m` represents the particle mass
- `V(x)` represents the scalar potential

---

## Features

### Quantum State Management

Provides utilities for:

- Two-component spinor construction
- Spinor normalisation (spatial or internal)
- Probability density calculation per component

### Solvers

Provides the `DiracSolver` class:

```python
DiracSolver(x, potential=None, mass=1.0, c=1.0, hbar=1.0)
```

| Method | Description |
|--------|-------------|
| `solve(psi0, t_end, dt)` | Evolve 2-component spatial spinor via matrix exponential |
| `solve_plane_wave(momentum, t_end, dt, spin_up)` | Free-particle plane-wave evolution |

### Results

The `DiracResult` object returned by the solver:

| Field | Description |
|-------|-------------|
| `spinor` | Spinor at each step `(n_steps, 2, n_grid)` |
| `probability` | Component probabilities `(n_steps, 2*n_grid)` |
| `t` | Time values |
| `x` | Spatial grid |
| `energy` | Computed energy spectrum |
| `spin_expectation` | `(⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)` per step |

**Properties:** `total_probability`, `upper_component`, `lower_component`

### Pauli and Dirac Matrices

```python
from usel.dirac import PAULI_X, PAULI_Y, PAULI_Z, PAULI_I
from usel.dirac import gamma_matrices_dirac, dirac_alpha_matrices, dirac_beta
```

| Object | Description |
|--------|-------------|
| `PAULI_X`, `PAULI_Y`, `PAULI_Z` | Standard 2×2 Pauli matrices |
| `PAULI_I` | 2×2 identity |
| `gamma_matrices_dirac()` | Returns (γ⁰, γ¹, γ², γ³) in standard representation |
| `dirac_alpha_matrices()` | Returns (α₁, α₂, α₃) |
| `dirac_beta()` | β = γ⁰ matrix |

### Spinor Utilities

```python
from usel.dirac import normalise_spinor, spin_expectation
```

- `normalise_spinor(spinor, dx)` — Normalise spatial or internal spinor
- `spin_expectation(spinor, pauli)` — Compute ⟨ψ|σ|ψ⟩

### Numerical Method

The solver uses the **matrix exponential** propagator:

\[
\psi(t + dt) = \exp(-iH\,dt/\hbar)\,\psi(t)
\]

computed via `scipy.linalg.expm` for the full 2N × 2N Dirac Hamiltonian on
the finite-difference grid.

---

## Module Structure

```text
dirac/

├── solver.py
├── matrices.py
├── spinors.py
├── propagators.py
├── validation.py
```

---

## Example

Run the free-particle example:

```bash
uv run example/dirac/example_free_particle.py
```

---

## Development

Install dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Run quality checks:

```bash
uv run ruff check .
uv run black --check .
uv run mypy src
```

---

## Testing

The Dirac module includes tests for:

- Matrix definitions (Pauli, Dirac, gamma matrices)
- Spinor normalisation and expectation values
- Plane-wave and spatial propagation
- Energy spectrum and spin conservation

Run:

```bash
uv run pytest tests/test_dirac
```

---

## Hardware Requirements

Minimum:

- CPU-based system
- 4 GB RAM

Recommended:

- Multi-core processor
- 8 GB RAM

GPU acceleration is not required.

---

## Validation

- **Free particle:** energy matches `E = √((pc)² + (mc²)²)`
- **Plane wave:** spin is conserved under free evolution
- **Spinor norm:** preserved under unitary propagation

---

## Future Development

Planned improvements:

- 2D and 3D relativistic wave-packet simulations
- Coupling to external electromagnetic fields
- Klein-Gordon equation module
- Research-oriented relativistic scattering workflows

---

## Contributing

Contributions are welcome.

Please ensure:

- Code follows PEP 8 standards
- Tests are added for new features
- Documentation is updated
- CI checks pass successfully

---

## License

Part of the **USEL (Universal Scientific Engine Library)** project.

Licensed under the MIT License.