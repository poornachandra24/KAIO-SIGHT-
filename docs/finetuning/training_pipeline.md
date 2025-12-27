# Training Pipeline

## Overview
The training pipeline is orchestrated by `scripts/run_train.sh`. It is designed to be a "Zero-Idle" system, ensuring that the GPU is constantly fed with data.

## Orchestration (`scripts/run_train.sh`)
The script follows a 3-stage process:

1.  **Stage 1: Audit & Ingestion**
    - Checks if the required raw data chunks are present.
    - Triggers `downloader.py` if data is missing.
2.  **Stage 2: Offline ETL (Anti-Bottleneck)**
    - Checks for pre-processed binary shards in `data/shards`.
    - Triggers `prepare_dataset.py` if shards are missing. This ensures heavy preprocessing is done *before* training starts.
3.  **Stage 3: Training**
    - Launches the main training script: `python3 -m src.training.trainer`.

## Data Loading (`src/training/sharded_dataset.py`)
To handle large datasets efficiently, we use a custom `ShardedDataset` class.

### Key Features
- **Streaming**: Inherits from `IterableDataset` to stream data from disk, keeping RAM usage low.
- **Lazy Loading**: Only loads data when iterated upon.
- **Multi-Worker**: Intelligently splits shards among PyTorch DataLoader workers.

## Training Configuration
Training parameters are defined in `configs/finetuning_config.yaml`. The system supports:
- **LoRA/QLoRA**: For efficient fine-tuning of large vision-language models.
- **Unsloth**: Integrated for faster training and memory efficiency.
