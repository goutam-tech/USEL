#!/usr/bin/env bash
# Run the full quality gate used before tagging a release.
set -euo pipefail

echo "Running Ruff..."
poetry run ruff check .

echo "Running Black (check mode)..."
poetry run black --check .

echo "Running MyPy..."
poetry run mypy src

echo "Running pytest with coverage..."
poetry run pytest --cov=usel --cov-report=term-missing

echo "All release checks passed."
