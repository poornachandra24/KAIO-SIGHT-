# ⚙️ System Configuration

The behavior of the KAIO-SIGHT system is controlled by YAML configuration files located in the `configs/` directory. This decoupling allows for easy tuning of hyperparameters and data settings without modifying the codebase.

## 1. Data Configuration (`configs/data_config.yaml`)
Controls the data ingestion and preprocessing pipeline.

```yaml
dataset:
  repo_id: "nvidia/PhysicalAI-Autonomous-Vehicles"
  index_file: "clip_index.parquet"
  local_dir: "/workspace/KAIO-SIGHT/data"

download:
  max_concurrent: 12
  # NEW: Primary control for data scaling
  target_chunks: 1 
  # Secondary control for disk safety
  target_size_gb: 150 

schema:
  time_col: "timestamp"
  translation_cols: ["x", "y", "z"]
  velocity_cols: ["vx", "vy"]
  rotation_cols: ["qx", "qy", "qz", "qw"]

processing:
  sequence_length: 16 
  stride: 8 

cameras:
  - "camera_front_wide_120fov"
  - "camera_front_tele_30fov"
  - "camera_rear_left_70fov"
  - "camera_rear_right_70fov"
  - "camera_cross_left_120fov"
  - "camera_cross_right_120fov"
  - "camera_rear_tele_30fov"
```

## 2. Finetuning Configuration (`configs/finetuning_config.yaml`)
Controls the model architecture and training hyperparameters.

```yaml
project_name: "KAIO-SIGHT Finetuning-BS-144"

hub:
  hf_repo: "Thunderbird2410/KAIO-SIGHT"

reporting:
  use_comet_ml: true

model:
  base_model: "Qwen/Qwen2.5-VL-7B-Instruct"
  max_seq_length: 65536  
  lora_rank: 128
  lora_alpha: 256
  # Resolution settings used inside trainer.py logic
  resolution: 504
  
  # LoRA Target Modules
  # Default (Small): ["q_proj", "k_proj", "v_proj", "o_proj"]
  # Expanded (Large): ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  # Batch Size Strategy for MI300X Stability
  # We force BS=1 in code to prevent broadcasting errors.
  # We use Grad Accumulation to reach the target effective batch size.
  batch_size: 48              # Reduced from 64 (99%) to aim for ~75% VRAM
  gradient_accumulation_steps: 3  # Effective Batch Size = 144
  
  # Optimization
  learning_rate: 1e-4
  optimizer: "paged_adamw_8bit"
  weight_decay: 0.01
  # Workers: Enable 8 CPU cores to prepare data in parallel
  dataloader_num_workers: 12 
  
  # Duration
  num_epochs: 1               # Set to 1 for the first production run; this covers the entire dataset.
  max_steps: -1              # -1 means "Run until num_epochs is finished"; else set a fixed number of steps.
  
  # Scheduling
  warmup_ratio: 0.1           # Warmup for 10% of total steps
  lr_scheduler_type: "cosine"  # Cosine learning rate decay
  
  # Logging & Saving
  logging_steps: 10            # Log loss every 5 steps
  save_steps: 10            # Save checkpoint every 500 steps (~45-60 mins)
  save_total_limit: 2         # Keep only the last 3 checkpoints to save disk space


vision:
  # Options: "4-cam" or "7-cam"
  camera_setup: "7-cam"
  #"4-cam"
  window_size: 16
  # The loader will look at these keys based on camera_setup
  setups:
    4-cam:
      grid: [2, 2]
      cameras: 
        - "camera_front_wide_120fov"
        - "camera_front_tele_30fov"
        - "camera_rear_left_70fov"
        - "camera_rear_right_70fov"
    7-cam:
      grid: [3, 3]
      cameras:
        - "camera_front_wide_120fov"
        - "camera_front_tele_30fov"
        - "camera_cross_left_120fov"
        - "camera_cross_right_120fov"
        - "camera_rear_left_70fov"
        - "camera_rear_right_70fov"
        - "camera_rear_tele_30fov"
```
