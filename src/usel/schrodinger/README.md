# Schrödinger Module

The Schrödinger module is a quantum mechanics component of **USEL
(Universal Scientific Engine Library)**.

It provides a modular computational framework for building and analyzing
non-relativistic quantum systems using numerical methods. The module is
designed for CPU-based scientific computing and works efficiently on standard
consumer hardware without GPU requirements.

---

## Overview

The module provides reusable components for quantum mechanical simulations:

- Quantum grid generation
- Potential models
- Mathematical operators
- Hamiltonian construction
- Quantum state evolution
- Observables calculation
- Validation utilities
- Visualization tools

The architecture is designed to support future extensions in computational
physics and quantum research.

---

## Mathematical Foundation

The module is based on the time-dependent Schrödinger equation:

\[
i\hbar\frac{\partial\psi}{\partial t}=H\psi
\]

where:

- `ψ` represents the quantum wavefunction
- `ℏ` represents the reduced Planck constant
- `H` represents the Hamiltonian operator

The Hamiltonian is represented as:

\[
H=T+V
\]

where:

- `T` represents kinetic energy
- `V` represents potential energy

---

## Features

### Quantum State Management

Provides utilities for:

- Complex wavefunctions
- State normalization
- Probability density calculation
- Quantum state validation

### Grid System

Supports:

- One-dimensional spatial grids
- Configurable resolution
- Coordinate generation
- Grid spacing calculation

### Potential Models

Includes:

- Infinite square well
- Harmonic oscillator
- Barrier potentials
- Custom user-defined potentials

### Operators

Provides:

- Position operators
- Momentum operators
- Kinetic energy operators
- Numerical differential operators

### Hamiltonian Construction

Supports construction of quantum Hamiltonians by combining:

- Kinetic operators
- Potential functions

### Time Evolution

Provides numerical evolution methods:

- Matrix exponential propagation
- Finite difference evolution
- Crank-Nicolson method
- Split operator method
- Runge-Kutta evolution
- Imaginary time evolution

### Observables

Supports:

- Probability density
- Expectation values
- Energy calculations
- State properties

### Boundary Conditions

Implemented:

- Dirichlet boundary conditions
- Periodic boundary conditions
- Absorbing boundary conditions

### Visualization

Provides:

- Wavefunction plots
- Probability density plots
- Potential visualization
- Time evolution visualization

---

## Module Structure

```

schrodinger/

├── grids.py
├── potentials.py
├── operators.py
├── hamiltonian.py

├── normalization.py
├── observables.py

├── boundary_conditions.py
├── evolution.py
├── propagators.py

├── validation.py
├── visualization.py

├── finite_difference.py
├── crank_nicolson.py
├── split_operator.py
├── imaginary_time.py
└── rk4.py

```

---

## Example

Run the Schrödinger particle-in-box example:

```bash
uv run examples/example_6_schrodinger_particle_in_box.py
```

Example output:

```
Infinite Square Well — Energy Eigenvalues

n   Computed     Analytical

1   4.895465     4.934802
2   19.581668    19.739209
3   44.058032    44.413220
4   78.323593    78.956835
5   122.377005   123.370055
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

The Schrödinger module includes tests for:

- Grid generation
- Potential models
- Operator construction
- Hamiltonian validation
- Normalization
- Time evolution
- Numerical accuracy

Run:

```bash
uv run pytest tests/test_schrodinger
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

## Future Development

Planned improvements:

- Two-dimensional quantum systems
- Three-dimensional quantum systems
- Advanced numerical solvers
- Quantum chemistry extensions
- Many-body quantum simulations
- Research-oriented workflows

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

```
This is the style expected for a real open-source scientific module repository. It matches your USEL structure and is suitable for GitHub.
```
