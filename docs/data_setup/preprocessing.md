# Data Preprocessing (Turbo ETL)

## Overview
To ensure high-throughput training on MI300X GPUs, we implement an "Offline ETL" strategy. Instead of processing raw video data on-the-fly (which can bottleneck the GPU), we pre-process the data into binary tensors.

## The Pipeline (`src/data/prepare_dataset.py`)
The `prepare_dataset.py` script implements a **GPU-Accelerated Turbo ETL** pipeline.

### Key Features
- **GPU Acceleration**: Uses PyTorch on ROCm to perform image resizing and tensor operations on the GPU.
- **Parallel Processing**: Uses `multiprocessing` to handle multiple data chunks concurrently.
- **Sharding**: Saves processed data into "micro-shards" (Hugging Face Datasets) to manage memory and enable efficient streaming.

### Steps
1.  **Extraction**: Unzips MP4 videos and Parquet labels to a temporary RAM disk (`/dev/shm`) for speed.
2.  **Frame Processing**:
    - Decodes video frames using `cv2`.
    - Uploads frames to GPU.
    - Resizes frames to the target resolution (e.g., 896x896).
    - **Tiling**: Combines 4 camera views (Front, Right, Left, Rear) into a single grid image.
3.  **Label Synchronization**: Aligns video frames with kinematic data (velocity, displacement) to create training samples.
4.  **Serialization**: Saves the processed samples (images + text instructions) to disk as binary shards.

## Output
The result is a set of binary shards in `data/processed_dataset` (or `data/shards` during intermediate steps). These shards are ready for direct streaming by the training loop.

## Handling Challenges
- **Arrow Overflow**: The script merges shards in batches to prevent Apache Arrow memory overflow issues.
- **VRAM Contention**: The number of workers is tuned to balance CPU decompression and GPU processing without OOM errors.
