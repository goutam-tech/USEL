# Contributing to USEL

Thanks for your interest in contributing! USEL is an open-source project
and welcomes contributions of all kinds: bug reports, bug fixes, new
features, documentation improvements, and tests.

## Getting Started

1. Fork the repository and clone your fork.
2. Run `./scripts/setup_dev_env.sh` to install dependencies and pre-commit
   hooks.
3. Create a feature branch: `git checkout -b feature/my-change`.

## Development Workflow

1. Make your changes in `src/usel/`.
2. Add or update tests in `tests/` (mirroring the package layout).
3. Run the full quality gate: `./scripts/run_release_checks.sh`.
4. Update documentation under `docs/` if you changed public behavior.
5. Update `CHANGELOG.md` under an "Unreleased" heading.

## Pull Request Guidelines

- Keep pull requests focused on a single change.
- Write clear commit messages.
- Ensure all CI checks pass (lint, format, type check, tests).
- Maintain or improve test coverage (target: 95%+).
- Follow the existing code style: PEP 8, full type hints, NumPy-style
  docstrings, composition over inheritance, no global mutable state.

## Reporting Issues

Please include:

- USEL version (`usel version`)
- Python version and OS
- A minimal reproducible example
- The full error traceback, if applicable

## Code of Conduct

Be respectful and constructive. This project follows the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
