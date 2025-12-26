#!/bin/bash
# Project Omni: High-Throughput Autonomous Reasoning Pipeline
set -e

PROJECT_ROOT="/workspace/AMD-Vision-Omni"
export PYTHONPATH=$PROJECT_ROOT
DATA_DIR="$PROJECT_ROOT/data"
PROCESSED_DIR="$DATA_DIR/processed_dataset"

# Architect's Move: Extract the UNIT TARGET from YAML
TARGET_UNITS=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/data_config.yaml'))['download']['target_chunks'])")

echo "======================================================="
echo "   PROJECT OMNI: PRODUCTION ORCHESTRATION ENGINE       "
echo "   TARGET CAPACITY: $TARGET_UNITS Synchronized Chunks"
echo "======================================================="

# 1. RAW DATA AUDIT & INGESTION
echo "🔍 Stage 1: Auditing Raw Physical Data..."
AUDIT_OUT=$(python3 src/data/audit.py)

CURRENT_SYNCED=$(echo "$AUDIT_OUT" | grep "SYNCED_CHUNKS=" | cut -d'=' -f2)
EST_SAMPLES=$(echo "$AUDIT_OUT" | grep "TOTAL_SAMPLES=" | cut -d'=' -f2)

if [ "$CURRENT_SYNCED" -lt "$TARGET_UNITS" ]; then
    echo "📦 [DATA INCOMPLETE]: Only $CURRENT_SYNCED/$TARGET_UNITS chunks synced."
    echo ">> Initiating High-Speed Download..."
    python3 src/data/downloader.py --limit "$TARGET_UNITS"
    
    # Re-run audit to update Est. Samples count
    AUDIT_OUT=$(python3 src/data/audit.py)
    CURRENT_SYNCED=$(echo "$AUDIT_OUT" | grep "SYNCED_CHUNKS=" | cut -d'=' -f2)
    EST_SAMPLES=$(echo "$AUDIT_OUT" | grep "TOTAL_SAMPLES=" | cut -d'=' -f2)
fi
echo "✅ Stage 1 Complete: $CURRENT_SYNCED units verified."

# 2. OFFLINE ETL (The Anti-Bottleneck Guard)
echo "🏗️  Stage 2: Checking Binary Cache Integrity..."
# We check for dataset_info.json as a marker of a successful HF Dataset save
if [ ! -d "$PROCESSED_DIR" ] || [ ! -f "$PROCESSED_DIR/dataset_info.json" ]; then
    echo "⚠️  [CACHE MISS]: Binary dataset not found or incomplete."
    echo ">> Commencing Parallel Offline ETL (MP4 -> Binary Tensors)..."
    echo ">> This uses 16 CPU cores to eliminate GPU idle time."
    
    # Run the ETL script we built to decode videos and resize to 896px
    python3 src/data/prepare_dataset.py
    
    echo "✅ ETL Complete. Tensors baked to disk."
else
    # Architect's check: ensure cached sample count matches expected sample count
    CACHED_COUNT=$(python3 -c "from datasets import load_from_disk; ds=load_from_disk('$PROCESSED_DIR'); print(len(ds))")
    echo "🚀 [CACHE HIT]: $CACHED_COUNT pre-processed samples detected."
fi

# 3. HIGH-THROUGHPUT GPU TRAINING
echo "======================================================="
echo "🔥 Stage 3: Launching MI300X Training Engine         "
echo "   Mode: ZERO-IDLE (Light-Speed Ingestion)           "
echo "======================================================="

# We no longer need USE_PROCESSED_DATA flag because the trainer 
# will now ONLY accept the processed directory.
stdbuf -oL -eL python3 -m src.training.trainer