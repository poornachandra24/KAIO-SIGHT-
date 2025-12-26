#!/bin/bash
set -e

PROJECT_ROOT="/workspace/AMD-Vision-Omni"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Architect's Move: Extract the UNIT TARGET from YAML
TARGET_UNITS=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/data_config.yaml'))['download']['target_chunks'])")

echo "-------------------------------------------------------"
echo "   PROJECT OMNI: UNIT-AWARE SYSTEM AUDIT               "
echo "   TARGET: $TARGET_UNITS Synchronized Chunks          "
echo "-------------------------------------------------------"

# 1. Run the Deep Data Audit
echo "🔍 Running Deep Data Audit (src/data/audit.py)..."
AUDIT_OUT=$(python3 src/data/audit.py)

# Extract metrics
CURRENT_SYNCED=$(echo "$AUDIT_OUT" | grep "SYNCED_CHUNKS=" | cut -d'=' -f2)
EST_SAMPLES=$(echo "$AUDIT_OUT" | grep "TOTAL_SAMPLES=" | cut -d'=' -f2)

# 2. Unit-Based Decision Logic
if [ "$CURRENT_SYNCED" -ge "$TARGET_UNITS" ]; then
    echo "✅ DATA STABLE: $CURRENT_SYNCED/$TARGET_UNITS chunks synced."
    echo "📈 Training Capacity: ~ $EST_SAMPLES temporal windows."
else
    echo "📦 DATA INCOMPLETE: Only $CURRENT_SYNCED/$TARGET_UNITS chunks are synced."
    echo ">> Triggering Slicer for $TARGET_UNITS units..."
    # We pass the limit directly to the downloader
    python3 src/data/downloader.py --limit "$TARGET_UNITS"
    
    # Re-verify after download
    AUDIT_OUT=$(python3 src/data/audit.py)
    CURRENT_SYNCED=$(echo "$AUDIT_OUT" | grep "SYNCED_CHUNKS=" | cut -d'=' -f2)
    echo "✅ Download Complete. Now synced: $CURRENT_SYNCED chunks."
fi

# 3. Check Processed Cache
PROCESSED_DIR="$PROJECT_ROOT/data/processed_dataset"
if [ -d "$PROCESSED_DIR" ] && [ -f "$PROCESSED_DIR/dataset_info.json" ]; then
    echo "🚀 PROCESSED CACHE: Found. Mode: LIGHT-SPEED."
    export USE_PROCESSED_DATA=1
else
    echo "⚠️ PROCESSED CACHE: Missing. Mode: JIT-STREAMING."
    export USE_PROCESSED_DATA=0
fi

echo "-------------------------------------------------------"
echo "🚀 Initializing Training Engine on MI300X...           "
echo "-------------------------------------------------------"

stdbuf -oL -eL python3 -m src.training.trainer