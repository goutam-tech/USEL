#!/usr/bin/env bash
# Bootstrap a local usel development environment.
set -euo pipefail

echo "Installing Poetry dependencies..."
poetry install

echo "Installing pre-commit hooks..."
poetry run pre-commit install

echo "Running test suite..."
poetry run pytest

echo "Done. Activate the environment with: poetry shell"
