# ⚙️ System Configuration

The behavior of the KAIO-SIGHT system is controlled by YAML configuration files located in the `configs/` directory. This decoupling allows for easy tuning of hyperparameters and data settings without modifying the codebase.

## 1. Data Configuration (`configs/data_config.yaml`)
Controls the data ingestion and preprocessing pipeline.

```yaml
dataset:
  repo_id: "nvidia/PhysicalAI-Autonomous-Vehicles"  # Hugging Face Source
  index_file: "clip_index.parquet"
  local_dir: "/workspace/KAIO-SIGHT/data"      # Mapped Volume

download:
  max_concurrent: 12  # Parallel downloads
  target_chunks: 1    # Scale factor (How many chunks to sync)
  target_size_gb: 150 # Safety limit

processing:
  sequence_length: 16 # Frames per sample
  stride: 8           # Overlap
```

## 2. Finetuning Configuration (`configs/finetuning_config.yaml`)
Controls the model architecture and training hyperparameters.

```yaml
model:
  base_model: "Qwen/Qwen2.5-VL-7B-Instruct"
  max_seq_length: 4096
  lora_rank: 64
  lora_alpha: 128

training:
  batch_size: 1               # Per device (BS=1 avoids broadcasting errors)
  gradient_accumulation_steps: 8 # Effective BS = 8
  learning_rate: 2e-4
  optimizer: "adamw_torch"
  
vision:
  camera_setup: "4-cam"       # Multi-view configuration
  window_size: 16             # Temporal window
```
