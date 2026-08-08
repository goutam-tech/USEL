# Black Scholes

The Black-Scholes module is a financial-mathematics component of **USEL
(Universal Scientific Engine Library)**.

It provides a modular, NumPy-vectorized implementation of the Black-Scholes
option pricing model — analytical formulas, Greeks, put-call parity,
implied volatility, the governing PDE, a finite-difference solver, and a
Monte Carlo simulation engine, all cross-validated against one another. The
module is designed for CPU-based scientific computing and works efficiently
on standard consumer hardware without GPU requirements.

---

## Overview

The module provides reusable components for option-pricing analysis:

- Closed-form analytical pricing
- Greeks and put-call parity
- Implied volatility solving
- Finite-difference PDE solving
- Monte Carlo simulation

Three independent pricing paths — closed-form, finite-difference PDE, and
Monte Carlo — are implemented so that each can validate the others.

---

## Mathematical Foundation

The module is based on the Black-Scholes PDE:

\[
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
\]

with the closed-form solution for a European call:

\[
C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2)
\]

where:

- `S` represents the underlying asset price
- `K` represents the strike price
- `r` represents the risk-free rate
- `q` represents the continuous dividend yield
- `σ` represents volatility
- `T` represents time to expiry
- `N(·)` represents the standard normal CDF

---

## Features

### Probability Distributions

`distributions.py` — standard normal PDF, log-PDF, CDF, survival function,
and inverse CDF (quantile function), vectorized over scalars and arrays.

### Core Parameters

`parameters.py` — `d1` and `d2`, with dividend yield support and
broadcasting.

### Analytical Pricing

`pricing.py` — closed-form European call and put prices, with correct
handling of `T = 0` (intrinsic value).

### Put-Call Parity

`parity.py` — recover one price from the other, compute the parity
residual, and verify consistency within a tolerance.

### Greeks

`greeks.py` — delta, gamma, vega, theta, and rho for both calls and puts,
each verified against finite-difference derivatives of the pricing
functions.

### Implied Volatility

`implied_volatility.py` — a safeguarded Newton-Raphson / bisection solver
with no-arbitrage bound validation.

### The Black-Scholes PDE

`pde.py` — the PDE residual (verified analytically using the closed-form
price and Greeks), terminal (payoff) conditions, and boundary conditions
as `S -> 0` and `S -> infinity`.

### Finite-Difference Solver

`finite_difference.py` — a Crank-Nicolson (default; also explicit and
fully implicit) grid solver for the Black-Scholes PDE, cross-validated
against the closed-form price.

### Monte Carlo Pricing

`stochastic.py` — risk-neutral GBM simulation with antithetic variance
reduction, confidence intervals, and cross-validation against the
closed-form and PDE solutions.

---

## Module Structure

```text
black_scholes/

├── distributions.py
├── parameters.py
├── pricing.py
├── parity.py
├── greeks.py
├── implied_volatility.py
├── pde.py
├── finite_difference.py
└── stochastic.py

tests/

├── test_distributions.py
├── test_parameters.py
├── test_pricing.py
├── test_parity.py
├── test_greeks.py
├── test_implied_volatility.py
├── test_pde.py
├── test_finite_difference.py
└── test_stochastic.py
```

---

## Installation

```bash
pip install usel
```

Or, for local development:

```bash
git clone https://github.com/<org>/usel.git
cd usel/src/black_scholes
pip install -e ".[dev]"
```

---

## Example

For the quick start check the example folder from the GitHub repo.

All functions accept either Python scalars or NumPy arrays and follow
NumPy broadcasting rules.

---

## Development

Install dependencies:

```bash
./scripts/setup_dev_env.sh
```

Run the full quality gate:

```bash
./scripts/run_release_checks.sh
```

---

## Testing

- `pytest`, with a 98% coverage target.
- Known-value regression tests pinned with `pytest.approx(..., rel=1e-10)`.
- Cross-validation between independent implementations wherever possible
  (e.g. analytical Greeks checked against finite differences of
  `pricing.py`; the finite-difference solver checked against the
  closed-form price; Monte Carlo checked against both).

Run:

```bash
uv run pytest tests/test_black_scholes
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

- Higher-order Greeks (vanna, volga, charm, speed)
- American option pricing (binomial and PSOR methods)
- Local and stochastic volatility model support
- Multi-asset basket option pricing

---

## Contributing

Thanks for your interest in contributing! `black_scholes` welcomes bug
reports, bug fixes, new features, documentation improvements, and tests.

### Getting Started

1. Fork the repository and clone your fork.
2. Run `./scripts/setup_dev_env.sh` to install dependencies and
   pre-commit hooks.
3. Create a feature branch: `git checkout -b feature/my-change`.

### Development Workflow

1. Make your changes in `black_scholes/`.
2. Add or update tests in `tests/` (mirroring the package layout).
3. Run the full quality gate: `./scripts/run_release_checks.sh`.
4. Update documentation under `docs/` if you changed public behavior.
5. Update `CHANGELOG.md` under an "Unreleased" heading.

### Pull Request Guidelines

- Keep pull requests focused on a single change.
- Write clear commit messages.
- Ensure all CI checks pass (lint, format, type check, tests).
- Maintain or improve test coverage (target: 95%+).
- Follow the existing code style: PEP 8, full type hints, NumPy-style
  docstrings, composition over inheritance, no global mutable state.
- For any new or modified pricing formula, cite the source (textbook
  section, paper, or reference site) in the docstring, in the same style
  as the existing modules.

### Reporting Issues

Please include:

- `black_scholes` version (`black_scholes.__version__`)
- Python version and OS
- A minimal reproducible example
- The full error traceback, if applicable

### Code of Conduct

Be respectful and constructive. This project follows the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).

---

## License

Part of the **USEL (Universal Scientific Engine Library)** project.

By contributing, you agree that your contributions will be licensed under
the MIT License.

---

## References

The formulas implemented in this library follow standard, well-established
references in financial mathematics. Contributors adding or modifying a
formula should cite the specific source in the corresponding docstring.

### Foundational Papers

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate
  Liabilities*. Journal of Political Economy, 81(3), 637–654. — the
  original derivation of the option pricing formula and the PDE
  (`pricing.py`, `pde.py`).
- Merton, R. C. (1973). *Theory of Rational Option Pricing*. The Bell
  Journal of Economics and Management Science, 4(1), 141–183. — extension
  of the model to include continuous dividend yield (`parameters.py`,
  `pricing.py`, `greeks.py` dividend-yield terms).

### Textbooks

- Hull, J. C. *Options, Futures, and Other Derivatives* (current
  edition). Pearson. — primary reference for the Greeks (`greeks.py`),
  put-call parity (`parity.py`), and the general treatment of the
  Black-Scholes-Merton model.
- Wilmott, P. *Paul Wilmott on Quantitative Finance* (current edition).
  Wiley. — reference for the Black-Scholes PDE, boundary/terminal
  conditions (`pde.py`), and finite-difference discretization schemes
  (`finite_difference.py`).
- Haug, E. G. *The Complete Guide to Option Pricing Formulas* (current
  edition). McGraw-Hill. — reference for the closed-form Greeks
  (including higher-order Greeks planned for future versions) and
  numerical implied-volatility solvers (`implied_volatility.py`).
- Glasserman, P. *Monte Carlo Methods in Financial Engineering*. Springer.
  — reference for risk-neutral GBM simulation and antithetic variance
  reduction (`stochastic.py`).
- Duffy, D. J. *Finite Difference Methods in Financial Engineering: A
  Partial Differential Equation Approach*. Wiley. — reference for the
  explicit, implicit, and Crank-Nicolson schemes and the Thomas algorithm
  used in `finite_difference.py`.

### Web References

- Investopedia — [Black-Scholes Model](https://www.investopedia.com/terms/b/blackscholes.asp) —
  accessible overview of the model and its assumptions.
- Wikipedia — [Black-Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) —
  formula summaries and derivation sketches, useful for cross-checking
  notation conventions.
- Wikipedia — [Greeks (finance)](https://en.wikipedia.org/wiki/Greeks_(finance)) —
  summary table of first- and second-order Greeks, referenced when
  extending `greeks.py`.
- CFA Institute curriculum materials on derivative pricing — referenced
  for standard notation and no-arbitrage bound conventions used in
  `implied_volatility.py`.