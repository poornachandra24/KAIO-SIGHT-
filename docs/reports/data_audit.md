# Data Audit Report

## Purpose
The data audit ensures that the local dataset matches the expected configuration before training begins.

## Audit Script (`src/data/audit.py`)
This script scans the `data/` directory and reports:
- **Synced Chunks**: Number of chunks fully downloaded.
- **Total Samples**: Estimated number of training samples.
- **Missing Files**: Any discrepancies between the index and local files.

## Sample Output
```text
SYNCED_CHUNKS=150
TOTAL_SAMPLES=33153
STATUS=READY
```
