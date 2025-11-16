#!/bin/bash
# Test Runner Script for Bonus Level

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BONUS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BONUS_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Run tests
echo "=========================================="
echo "Running Bonus Level Tests"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "⚠ pytest not found. Installing test dependencies..."
    pip install -r "$BONUS_DIR/requirements-test.txt"
fi

# Run tests
pytest "$BONUS_DIR/tests/" -v --tb=short

# Optionally run with coverage
if [ "$1" == "--coverage" ] || [ "$1" == "-c" ]; then
    echo ""
    echo "=========================================="
    echo "Running Tests with Coverage"
    echo "=========================================="
    pytest "$BONUS_DIR/tests/" \
        --cov="$BONUS_DIR/core" \
        --cov="$BONUS_DIR/training" \
        --cov="$BONUS_DIR/evaluation" \
        --cov-report=html \
        --cov-report=term-missing
    echo ""
    echo "Coverage report generated in htmlcov/"
fi

echo ""
echo "=========================================="
echo "Tests Complete"
echo "=========================================="

