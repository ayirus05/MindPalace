#!/usr/bin/env bash
# MindPalace Tier 2 — test runner.
#
# This script installs dependencies and runs the full test suite with
# coverage.  Run it from the project root:
#
#   ./run_tests.sh
#
# Requirements: Python 3.10+, uv (recommended) or pip.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
if command -v uv &>/dev/null; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi

echo ""
echo "=== Running tests with coverage ==="
python -m pytest tests/ --cov=src/palace --cov-report=term-missing --cov-report=html "$@"

echo ""
echo "=== Test run complete ==="
echo "HTML coverage report: htmlcov/index.html"
