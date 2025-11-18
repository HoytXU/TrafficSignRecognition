#!/bin/bash
# Batch Training Script
# Wrapper script that calls the Python training script

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BONUS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BONUS_DIR")"

cd "$PROJECT_ROOT"

# Use Python script for better cross-platform support
python "$BONUS_DIR/scripts/train_all_models.py" "$@"

