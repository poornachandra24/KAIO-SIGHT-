# Data Audit Report

## Purpose
The data audit ensures that the local dataset matches the expected configuration before training begins.

## Audit Script (`src/data_etl/audit.py`)
This script scans the `data/` directory and reports:
- **Synced Chunks**: Number of chunks fully downloaded.
- **Total Samples**: Estimated number of training samples.
- **Missing Files**: Any discrepancies between the index and local files.

## Sample Output
```
# Physical AI Data Audit: 2025-12-26 16:42
- **Data Root**: `/workspace/KAIO-SIGHT/data`
- **Target Configuration**: `4-cam` (4 cameras) 

## Integrity Summary
- **Chunks Found**: 1
- **Fully Synchronized Chunks**: 1
- **Orphaned/Incomplete Chunks**: 0

## Training Capacity
- **Estimated Total Training Samples**: 276
- **Total Synchronized Data Size**: 7.67 GB
```
