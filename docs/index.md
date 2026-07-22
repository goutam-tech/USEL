# usel

usel is an open-source, CPU-only scientific computing framework focused
on quantum physics simulations and numerical methods.

Version 1.0 establishes the mathematical and computational foundation for
future quantum mechanics, computational physics, and quantum information
modules, while remaining lightweight enough to run on standard consumer
hardware within 8 GB of RAM, with no GPU required.

## Why usel?

- **Reusable scientific core** — matrices, linear algebra, solvers, and
  transforms built on NumPy and SciPy.
- **Reproducible** — seeded random generation and deterministic numerical
  routines.
- **Extensible** — a modular architecture designed to support quantum
  mechanics, computational physics, and quantum information engines in
  future releases without breaking the core API.
- **Offline and private** — no telemetry, no network communication, no data
  collection.

## Installation

```bash
pip install usel
```

## Next Steps

- Read the [Architecture](architecture.md) overview
- Browse the [API Reference](api/math.md)
- See the [Developer Guide](developer_guide.md) to set up a local environment
