#!/bin/bash
# 03_inference.sh
# Logic: Load -> Predict
set -e

# Dynamic Project Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=$PROJECT_ROOT

echo "======================================================="
echo "   PROJECT OMNI: INFERENCE PROTOCOL                    "
echo "======================================================="

cd "$PROJECT_ROOT"

echo "🔮 Running Inference..."
python3 src/inference/test_inference.py

echo "🎉 Inference Complete."
