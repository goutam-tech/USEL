# Changelog

All notable changes to USEL are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Numerical-correctness test baseline for core scientific methods
  (`tests/src/math/`, `tests/src/solvers/`, `tests/src/linalg/`).
- Reproducible CPU and memory benchmark suite (`benchmarks/bench_suite.py`).
- Solver convergence diagnostics: `residual` field on
  `SolverResult` for root-finding solvers.
- Infinite-well analytical validation tests for the Schrödinger solver.

### Changed
- CI test matrix now uses only declared Python versions (3.11, 3.12).

### Fixed
- None yet.

### Removed
- None yet.

## [1.0.9] - 2026-10-20

### Added
- Initial public release of USEL v1.0.9.

---

## Version source

The authoritative version is declared in `pyproject.toml` under
`[project] version`. All release artifacts, changelog entries, and
documentation must reference this single source. Do not maintain a
separate version number elsewhere.

## Release procedure

1. Update `pyproject.toml` with the new version (SemVer).
2. Add a changelog entry under a new version section, using the
   `Added` / `Changed` / `Fixed` / `Removed` categories. Only document
   changes that are implemented and tested — never advertise planned
   features as available.
3. Run the full test suite (`uv run pytest`) and lint/format checks
   (`uv run ruff check .`, `uv run ruff format --check .`).
4. Build release artifacts (`uv build`) and inspect `dist/` to confirm
   expected files and metadata are present.
5. Tag the commit with the version number (`git tag -a v1.0.9 -m "..."`)
   and push the tag. The release workflow is triggered by the tag push,
   not by ordinary pull requests. Publishing is never automatic.

## Version consistency

Package metadata version (`pyproject.toml`) must match the git tag and
changelog header. Before releasing, verify:

```bash
# Check the declared version
python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"

# Build and inspect artifacts
uv build
ls dist/
```
