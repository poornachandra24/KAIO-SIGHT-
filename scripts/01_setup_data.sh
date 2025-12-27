#!/bin/bash
# 01_setup_data.sh
# Logic: Download -> Audit -> ETL -> Cache
set -e

# Dynamic Project Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=$PROJECT_ROOT

DATA_DIR="$PROJECT_ROOT/data"
SHARDS_DIR="$DATA_DIR/shards"

echo "======================================================="
echo "   PROJECT OMNI: DATA SETUP PROTOCOL                   "
echo "======================================================="

# 1. RAW DATA AUDIT & INGESTION
echo "🔍 Stage 1: Auditing Raw Physical Data..."
# Ensure we are in project root for python imports to work if needed, or use PYTHONPATH
cd "$PROJECT_ROOT"

# Architect's Move: Extract the UNIT TARGET from YAML
TARGET_UNITS=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/data_config.yaml'))['download']['target_chunks'])")
echo "   TARGET CAPACITY: $TARGET_UNITS Synchronized Chunks"

AUDIT_OUT=$(python3 src/data_etl/audit.py)
CURRENT_SYNCED=$(echo "$AUDIT_OUT" | grep "SYNCED_CHUNKS=" | cut -d'=' -f2)

if [ -z "$CURRENT_SYNCED" ]; then
    echo "⚠️  Could not parse audit output. Assuming 0 synced."
    CURRENT_SYNCED=0
fi

if [ "$CURRENT_SYNCED" -lt "$TARGET_UNITS" ]; then
    echo "📦 [DATA INCOMPLETE]: Only $CURRENT_SYNCED/$TARGET_UNITS chunks synced."
    echo ">> Initiating High-Speed Download..."
    python3 src/data_etl/downloader.py --limit "$TARGET_UNITS"
    
    # Re-run audit
    AUDIT_OUT=$(python3 src/data_etl/audit.py)
    CURRENT_SYNCED=$(echo "$AUDIT_OUT" | grep "SYNCED_CHUNKS=" | cut -d'=' -f2)
fi
echo "✅ Stage 1 Complete: $CURRENT_SYNCED units verified."

# 2. OFFLINE ETL (The Anti-Bottleneck Guard)
echo "🏗️  Stage 2: Checking Binary Cache Integrity..."

# Check if shards directory exists and has content
if [ ! -d "$SHARDS_DIR" ] || [ -z "$(ls -A $SHARDS_DIR 2>/dev/null)" ]; then
    echo "⚠️  [CACHE MISS]: Binary shards not found."
    echo ">> Commencing Parallel Offline ETL (MP4 -> Binary Tensors)..."
    echo ">> This uses 16 CPU cores to eliminate GPU idle time."
    
    # Run the ETL script
    python3 src/data_etl/prepare_dataset.py
    
    echo "✅ ETL Complete. Tensors baked to disk."
else
    echo "🚀 [CACHE HIT]: Pre-processed shards detected in $SHARDS_DIR."
fi

echo "🎉 Data Setup Complete."
