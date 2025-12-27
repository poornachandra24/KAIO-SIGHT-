# Data Sourcing and Downloading

## Overview
The **AMD-Vision-Omni** project utilizes the `nvidia/PhysicalAI-Autonomous-Vehicles` dataset hosted on Hugging Face. This dataset provides rich, multi-view video data and kinematic labels for autonomous vehicle training.

## Configuration
The data source is configured in `configs/data_config.yaml`. Key parameters include:
- **repo_id**: `nvidia/PhysicalAI-Autonomous-Vehicles`
- **index_file**: `clip_index.parquet`
- **local_dir**: `/workspace/AMD-Vision-Omni/data` (mapped inside the container)
- **target_chunks**: Controls the scale of data to download.

## The Downloader (`src/data/downloader.py`)
The `downloader.py` script is responsible for efficiently fetching data from Hugging Face.

### Workflow
1.  **Index Retrieval**: Downloads `clip_index.parquet` to understand the dataset structure.
2.  **Chunk Identification**: Identifies unique data "chunks" required for the target scale.
3.  **Filtered Download**:
    - Downloads **Camera Data**: `.zip` files containing MP4 video clips for multiple views (Front, Rear, Left, Right).
    - Downloads **Label Data**: `.zip` files containing Parquet files with kinematic data (position, velocity, etc.).
    - Uses `huggingface_hub` with `HF_HUB_ENABLE_HF_TRANSFER=1` for high-speed transfer.

## Usage
The download is typically triggered automatically by the `run_train.sh` orchestration script if the local data is insufficient.

```bash
# Manual trigger (inside container)
python3 src/data/downloader.py --limit 10
```
