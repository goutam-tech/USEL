# Developer Guide

## Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/) for dependency management

## Setup

```bash
git clone https://github.com/goutam-tech/usel.git
cd usel
./scripts/setup_dev_env.sh
```

This installs all dependencies (including dev tools), installs pre-commit
hooks, and runs the test suite.

## Common Tasks

Run tests:

```bash
poetry run pytest
```

Run tests with coverage:

```bash
poetry run pytest --cov=usel --cov-report=term-missing
```

Lint and format:

```bash
poetry run ruff check . --fix
poetry run black .
```

Type check:

```bash
poetry run mypy src
```

Run the full release quality gate:

```bash
./scripts/run_release_checks.sh
```

Serve documentation locally:

```bash
poetry run mkdocs serve
```

## Coding Standards

- PEP 8 formatting, enforced by Black and Ruff
- Full type hints, enforced by MyPy in strict mode
- NumPy-style docstrings on all public functions and classes
- SOLID principles and composition over inheritance
- No global mutable state (with the narrow exception of output-precision
  configuration)
- Pure functions preferred where practical

## Testing

- Unit tests live in `tests/` and mirror the `src/usel/` package layout.
- Target coverage is 95%+.
- Tests should cover both the success path and expected failure modes
  (raised exceptions).
