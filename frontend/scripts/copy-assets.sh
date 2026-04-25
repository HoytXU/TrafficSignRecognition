#!/usr/bin/env bash
# Copies repository assets into frontend/public/ so Vite can bundle them.
# Must be run from the frontend/ directory (npm run copy-assets handles this).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$FRONTEND_DIR")"

echo "Copying assets..."

mkdir -p "$FRONTEND_DIR/public/images/meta"

# 43 GTSRB class reference images
cp "$REPO_DIR/datasets/dataset2/Meta/"*.png "$FRONTEND_DIR/public/images/meta/"
echo "  ✓ 43 GTSRB class reference images → public/images/meta/"

# Analysis charts (real outputs from training)
cp "$REPO_DIR/bonus/analysis/model_comparison.png" "$FRONTEND_DIR/public/images/"
cp "$REPO_DIR/bonus/analysis/data_analysis.png"    "$FRONTEND_DIR/public/images/"
echo "  ✓ model_comparison.png + data_analysis.png → public/images/"

echo "Done. Run 'npm run dev' or 'npm run build'."
