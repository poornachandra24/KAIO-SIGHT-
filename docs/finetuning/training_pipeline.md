# Training Pipeline

## Overview
## Overview
The training pipeline is orchestrated by `scripts/02_finetune.sh`. It is designed to be a "Zero-Idle" system, ensuring that the GPU is constantly fed with data.

## Configuration
The pipeline is controlled by `configs/finetuning_config.yaml`. This file defines the model parameters, learning rates, and batch strategies. See [Configuration Docs](../architecture/configuration.md) for details.

## Orchestration (`scripts/02_finetune.sh`)
The training process is now decoupled from data setup.

1.  **Stage 1: Training**
    - Launches the main training script: `python3 -m src.training.trainer`.
    - Uses the pre-processed binary shards from `data/shards`.
2.  **Stage 2: Push to Hub**
    - Automatically uploads the fine-tuned LoRA adapters to Hugging Face using `scripts/push_model.py`.

## Usage

```bash
bash scripts/02_finetune.sh
```

## Data Loading (Map-Style Strategy)
To handle large datasets efficiently while avoiding the "IterableDataset Slicing Bug" (see [ADR-002](../reports/decision_log.md)), we use a **Map-Style Dataset** approach.

### Key Features
- **Mechanism**: Uses `datasets.load_from_disk` to load binary shards.
- **Zero-Copy**: Leverages **Memory Mapping (`mmap`)** to access data directly from disk without loading it all into RAM.
- **Consolidation**: Stitches multiple shards into a single virtual dataset using `concatenate_datasets`.

## Training Configuration
Training parameters are defined in `configs/finetuning_config.yaml`. The system supports:
- **LoRA/QLoRA**: For efficient fine-tuning of large vision-language models.
- **Unsloth**: Integrated for faster training and memory efficiency.
