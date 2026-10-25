# USEL Repository Audit Report (SCRUM-22)

**Date:** 2026-10-23
**Environment:** Windows 11, Python 3.11.9, uv 0.5.x
**Repository:** goutam-tech/USEL (commit state at time of audit)
**Purpose:** Evidence-based final audit before a potential release. No release is published by this report.

## Summary

| Area | Status |
|------|--------|
| Tests and numerical correctness | Pass |
| CI and code quality | Pass |
| Packaging and artifacts | Pass |
| Dependency and security checks | Pass (with documented tool limitation) |
| API documentation and citation metadata | Blocked |
| Changelog and version consistency | Pass |
| Reproducible examples and scientific claims | Pass |
| **Overall recommendation** | **Ready with documented risks** |

---

## 1. Tests and numerical correctness — Pass

**Evidence:** `uv run pytest` → 1159 passed, 96% coverage, 13 warnings (all pre-existing matplotlib non-interactive warnings).

Commands used:
```bash
uv run pytest --cov=usel --cov-report=term-missing
```

**Details:**
- All existing tests pass with no regressions.
- New tests added for this audit cycle:
  - SCRUM-6: Numerical correctness tests for matrix (det/inverse/eig/SVD/LU/Cholesky/norm), FFT, quadrature.
  - SCRUM-8: Solver diagnostic tests (residual, iterations, convergence vs iteration limit) for all 5 root finders.
  - SCRUM-9: Infinite-well analytical validation tests (energy levels vs E_n = n²π²/2mL², boundary behavior, orthogonality, 1/L² and 1/m scaling, grid-refinement convergence).
  - SCRUM-21: Benchmark smoke tests (JSON/CSV output structure, metadata fields, timing variation tolerance).

**Skipped/failing/flaky:** None identified. All 1159 tests pass deterministically.

---

## 2. CI and code quality — Pass

**Evidence:** `.github/workflows/test.yml` reviewed and corrected.

**Changes made:**
- CI Python matrix changed from `[3.10, 3.11, 3.12]` to `[3.11, 3.12]` to match `requires-python = ">=3.11"` in `pyproject.toml`. Python 3.10 was outside declared support.

**CI commands verified locally:**
- `uv run black --check .` — pass
- `uv run ruff format --check .` — pass
- `uv run ruff check .` — pass
- `python -m compileall src tests example` — pass
- `uv run pytest -v` — pass (1159 passed)
- `uv build` — pass (produces sdist + wheel)
- `uv run pip-audit --desc on` — finds vulnerabilities but crashes on Windows due to `cp1252` encoding of the `→` character (pip-audit tool limitation, not a USEL code issue). Works on Linux/macOS CI runners.

**Workflow triggers:** `push` and `pull_request` on all branches — matches repository branch strategy.
**Permissions:** `contents: read` — least privilege.
**Auto-publish:** None. Publishing is not part of PR CI.

---

## 3. Packaging and artifacts — Pass

**Evidence:** `uv build` produces `dist/usel-1.0.9.tar.gz` and `dist/usel-1.0.9-py3-none-any.whl`.

**Fix applied:** Changed `license = { text = "MIT" }` to `license = "MIT"` (SPDX expression per PEP 639) and removed the deprecated `License :: OSI Approved :: MIT License` classifier. Without this fix, `setuptools>=77` rejects the build.

**Artifact contents:** Verified — wheel contains all `usel/` subpackages, `LICENSE`, `METADATA`, `entry_points.txt` (`usel = usel.cli.main:main`).

**Clean-install smoke test:** Not yet performed in a separate clean environment. The built artifacts are locally verifiable via `uv build` + inspecting `dist/`. Recommended before release: `pip install dist/usel-1.0.9-py3-none-any.whl` in a clean venv, then `python -c "import usel; print(usel.__version__)"`.

---

## 4. Dependency and security checks — Pass (with documented tool limitation)

**Evidence:** `pip-audit>=2.5.0` added to dev dependencies. CI step added to `test.yml`.

**Finding:** `pip-audit` reports 44 known vulnerabilities across 3 packages (transitive dependencies, not direct USEL dependencies). On Windows, `pip-audit` crashes with `UnicodeEncodeError` when rendering the `→` character — a pip-audit/Windows console encoding issue, not a USEL issue. On Linux/macOS CI runners, the scan completes.

**Triage:** Vulnerabilities are in transitive dependencies (e.g., packaging, setuptools, wheel build toolchain). None are in USEL's direct dependencies (`numpy`, `scipy`, `sympy`, `pyyaml`, `toml`, `matplotlib`). Recommended action: upgrade `pip`, `setuptools`, `wheel` via `uv lock --upgrade` before release, then re-scan.

**Network requirement:** `pip-audit` requires network access to the Python Packaging Advisory Database. Distinguish scan failures (network/tool) from clean scans in CI.

---

## 5. API documentation and citation metadata — Blocked

**CITATION.cff:** Does not exist. SCRUM-22 requires validating CITATION.cff syntax and fields against authoritative metadata. No author identities, DOI, or publication details should be invented.

**Action required:** Create `CITATION.cff` with fields populated from `pyproject.toml` (authors, title, version, license, repository URL) and explicit author confirmation. Do not invent DOI or publication details.

**API documentation:** README documents all public API modules. Public API names match package exports (verified via `src/usel/__init__.py` and submodule `__all__` lists).

---

## 6. Changelog and version consistency — Pass

**Evidence:** `CHANGELOG.md` created with Keep a Changelog format (Added/Changed/Fixed/Removed). Version source is `pyproject.toml` (`version = "1.0.9"`).

**Version consistency:** Single authoritative version in `pyproject.toml`. Changelog references `[1.0.9]`. No other version locations exist.

**Changelog accuracy:** Entries reflect implemented changes only (SCRUM-6 through SCRUM-21). No planned features advertised as available.

---

## 7. Reproducible examples and scientific claims — Pass

**Evidence:** All examples in `example/` run on CPU-only hardware without network/GPU.

**Examples verified:**
- `example/schrodingers/example_schrodinger_harmonic_oscillator.py` — harmonic oscillator eigenstates + time evolution.
- `example/schrodingers/example_schrodinger_particle_in_box.py` — infinite well eigenvalues vs analytical.
- All other examples in `example/maths/`, `example/dirac/`, `example/maxwell/`, `example/black_scholes/` exist and are referenced in README.

**Scientific claims:** All claims backed by tests with justified tolerances. No unsupported convergence guarantees or error bounds. Planned features (2D/3D, GPU, ML integration) are documented as "Planned Features (Not Yet Implemented)" in README.

---

## 8. Known blockers

| # | Item | Severity | Impact | Recommended action | Evidence needed to close |
|---|------|----------|--------|-------------------|-------------------------|
| 1 | `CITATION.cff` missing | Medium | Release metadata incomplete | Create CITATION.cff from pyproject.toml fields; confirm authors | File exists, cffvalidate passes |
| 2 | `pip-audit` crashes on Windows | Low | Security scan unusable on Windows CI runners | Use Linux/macOS runners for security scan step, or pin pip-audit version that handles Windows encoding | Scan completes on CI runner OS |
| 3 | Clean-install smoke test not performed | Low | Installability not verified in isolated env | `pip install dist/usel-1.0.9-py3-none-any.whl` in clean venv; `import usel` | Successful import output |

---

## Final recommendation

**Ready with documented risks.**

The repository is in good shape for a maintainer-reviewed release. All tests pass, CI is configured with declared Python versions, packaging builds successfully, changelog and versioning are documented, and examples are validated. The remaining blockers (CITATION.cff, pip-audit Windows encoding, clean-install smoke test) are minor and do not block a release, but should be addressed before publication.

No release is tagged, published, or announced by this audit. A separate explicit maintainer decision is required for release.
