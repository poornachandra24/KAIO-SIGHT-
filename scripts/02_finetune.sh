#!/bin/bash
# 02_finetune.sh
# Logic: Train -> Save -> Push
set -e

# Dynamic Project Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=$PROJECT_ROOT

echo "======================================================="
echo "   PROJECT OMNI: FINETUNE PROTOCOL                     "
echo "======================================================="

cd "$PROJECT_ROOT"

# 1. HIGH-THROUGHPUT GPU TRAINING
echo "🔥 Stage 1: Launching MI300X Training Engine"
echo "   Mode: ZERO-IDLE (Light-Speed Ingestion)"

# Launch Trainer
stdbuf -oL -eL python3 -m src.training.trainer

echo "✅ Training Complete."

# 2. PUSH TO HUB
echo "🚀 Stage 2: Pushing Model to Hub..."
python3 scripts/push_model.py

echo "🎉 Finetune & Push Complete."
