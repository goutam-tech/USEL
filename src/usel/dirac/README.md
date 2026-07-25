# Dirac Equation Module

Numerical solvers for the 1D Dirac equation in relativistic quantum mechanics, with spinor algebra and Pauli/Dirac matrix definitions.

## Overview

The Dirac equation describes relativistic spin-½ particles:

```
iℏ ∂ψ/∂t = H_D ψ
```

where the Dirac Hamiltonian is:

```
H_D = c α·p + β m c² + V(x)
```

This module provides:

- **Free-particle** plane-wave spinor evolution
- **Spatial** Dirac wave-packet propagation
- Complete Pauli and Dirac matrix library
- Spinor normalisation and expectation values

## Quick Start

```python
from usel.dirac import DiracSolver
import numpy as np

# Free particle plane wave
solver = DiracSolver(x=np.linspace(-5, 5, 100))
result = solver.solve_plane_wave(momentum=1.0, t_end=5.0, dt=0.01)

print(result.energy)             # E = sqrt((pc)² + (mc²)²)
print(result.spin_expectation)   # ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ over time
```

## API Reference

### `DiracSolver`

**Constructor:**
```python
DiracSolver(x, potential=None, mass=1.0, c=1.0, hbar=1.0)
```
- `x` — 1D spatial grid
- `potential` — Scalar potential V(x) (default: free particle)
- `mass` — Particle mass
- `c` — Speed of light
- `hbar` — Reduced Planck constant

**Methods:**

| Method | Description |
|--------|-------------|
| `solve(psi0, t_end, dt)` | Evolve 2-component spatial spinor via matrix exponential |
| `solve_plane_wave(momentum, t_end, dt, spin_up)` | Free-particle plane-wave evolution |

### `DiracResult`

| Field | Description |
|-------|-------------|
| `spinor` | Spinor at each step `(n_steps, 2, n_grid)` |
| `probability` | Component probabilities `(n_steps, 2*n_grid)` |
| `t` | Time values |
| `x` | Spatial grid |
| `energy` | Computed energy spectrum |
| `spin_expectation` | `(⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)` per step |

**Properties:** `total_probability`, `upper_component`, `lower_component`

## Pauli and Dirac Matrices

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

## Spinor Utilities

```python
from usel.dirac import normalise_spinor, spin_expectation
```

- `normalise_spinor(spinor, dx)` — Normalise spatial or internal spinor
- `spin_expectation(spinor, pauli)` — Compute ⟨ψ|σ|ψ⟩

## Numerical Method

The solver uses the **matrix exponential** propagator:

```
ψ(t + dt) = exp(-i H dt / ℏ) ψ(t)
```

computed via `scipy.linalg.expm` for the full 2N × 2N Dirac Hamiltonian on the finite-difference grid.

## Examples

See `examples/example_9_dirac_free_particle.py`.

## Validation

- **Free particle:** energy matches `E = √((pc)² + (mc²)²)`
- **Plane wave:** spin is conserved under free evolution
- **Spinor norm:** preserved under unitary propagation
