# AMD-Vision-Omni 🚀

<div align="center">

**A high-throughput vision-language model fine-tuning pipeline for autonomous vehicle reasoning**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%206.4-orange.svg)](https://rocm.docs.amd.com/)

</div>

---

## 📋 Overview

AMD-Vision-Omni is an end-to-end training pipeline designed to fine-tune vision-language models (VLMs) for autonomous driving tasks. The project leverages **AMD MI300X GPUs** with ROCm, enabling high-speed training on multi-camera video sequences from the NVIDIA PhysicalAI Autonomous Vehicles dataset.

### Key Features

- 🎬 **Multi-Camera Temporal Processing** - Processes 4-cam or 7-cam setups with dynamic spatial tiling
- 🧠 **Reasoning-Augmented Training** - Uses chain-of-thought style `<think>...action` formatting for enhanced reasoning
- ⚡ **Zero-Idle GPU Utilization** - Offline ETL pipeline pre-processes video data to binary tensors, eliminating runtime bottlenecks
- 🔧 **4-bit Quantized LoRA** - Efficient fine-tuning using Unsloth + PEFT with minimal VRAM footprint
- 📊 **Automated Data Pipeline** - Download, audit, process, and train with a single command

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  NVIDIA AV Dataset → Downloader → Raw Video → ETL → Binary Cache│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  Binary Cache → DataLoader → Qwen3-VL-8B → LoRA → Checkpoint    │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture

| Component | Details |
|-----------|---------|
| Base Model | `Qwen3-VL-8B-Instruct` (4-bit quantized via BitsAndBytes) |
| Fine-tuning | LoRA with rank=64, alpha=64 on Q/K/V/O projections |
| Max Sequence | 65,536 tokens |
| Vision Encoder | Native Qwen VL vision tower |

---

## 📂 Project Structure

```
AMD-Vision-Omni/
├── configs/
│   ├── data_config.yaml        # Dataset and download settings
│   └── finetuning_config.yaml  # Model and training hyperparameters
├── data/
│   └── processed_dataset/      # Cached HuggingFace dataset (auto-generated)
├── scripts/
│   └── run_train.sh            # Main orchestration script
├── src/
│   ├── data/
│   │   ├── audit.py            # Data integrity verification
│   │   ├── downloader.py       # Parallel chunk downloader
│   │   ├── loader.py           # PyTorch dataset with multi-cam tiling
│   │   ├── prepare_dataset.py  # Offline CPU ETL pipeline
│   │   └── processor.py        # Video frame extraction utilities
│   ├── models/
│   │   └── qwen_vl_arch.py     # Model initialization helpers
│   └── training/
│       ├── compute.py          # AMD GPU compute configuration
│       └── trainer.py          # SFTTrainer with custom logging
├── checkpoints/                # Training outputs
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- AMD GPU with ROCm 6.4+ (tested on MI300X)
- ~150GB disk space for dataset

### Setup

```bash
# Clone the repository
git clone https://github.com/poornachandra24/AMD-Vision-Omni.git
cd AMD-Vision-Omni

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` includes PyTorch with ROCm support. For NVIDIA GPUs, modify the torch installation accordingly.

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
bash scripts/run_train.sh
```

This will:
1. **Audit** - Check data integrity and download status
2. **Download** - Fetch missing data chunks (configurable in `data_config.yaml`)
3. **ETL** - Pre-process videos to binary tensors (runs once, cached)
4. **Train** - Launch LoRA fine-tuning on MI300X

### Option 2: Step-by-Step

```bash
# 1. Audit existing data
python src/data/audit.py

# 2. Download data chunks
python src/data/downloader.py --limit 5

# 3. Pre-process to binary cache
python src/data/prepare_dataset.py

# 4. Train the model
python -m src.training.trainer
```

---

## ⚙️ Configuration

### Data Configuration (`configs/data_config.yaml`)

```yaml
dataset:
  repo_id: "nvidia/PhysicalAI-Autonomous-Vehicles"
  local_dir: "/workspace/AMD-Vision-Omni/data"

download:
  target_chunks: 1      # Number of data chunks to download
  target_size_gb: 150   # Disk space limit

processing:
  sequence_length: 16   # Frames per training sample
  stride: 8             # Sliding window stride
```

### Training Configuration (`configs/finetuning_config.yaml`)

```yaml
model:
  base_model: "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
  max_seq_length: 65536
  lora_rank: 64
  lora_alpha: 64

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  num_epochs: 2
  optimizer: "paged_adamw_8bit"

vision:
  camera_setup: "4-cam"  # Options: "4-cam" or "7-cam"
  window_size: 16
```

---

## 📊 Dataset

This project uses the [NVIDIA PhysicalAI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset, which provides:

- Multi-camera video streams (7 cameras, various FOVs)
- Egomotion labels (position, velocity, rotation)
- High-quality driving scenarios

### Camera Configurations

| Setup | Cameras | Grid Layout |
|-------|---------|-------------|
| 4-cam | Front Wide, Front Tele, Rear Left, Rear Right | 2×2 |
| 7-cam | 4-cam + Cross Left, Cross Right, Rear Tele | 3×3 |

---

## 🔬 Training Output Format

The model is trained to produce structured reasoning outputs:

```
<think>
Avg Velocity: 5.23m/s. Total displacement: 42.15m.
</think>
Action: Displacement of 42.1 meters.
```

---

## 📈 Performance Notes

- **VRAM Usage**: ~40-60GB on MI300X with 4-bit quantization
- **Data Loading**: Near-zero GPU idle time with pre-processed binary cache
- **Throughput**: Optimized for high-speed training with gradient accumulation

---

## 🛣️ Roadmap

- [ ] Multi-GPU distributed training support
- [ ] Additional VLM backbones (LLaVA, InternVL)
- [ ] Inference pipeline with real-time camera input
- [ ] Extended temporal reasoning (>16 frames)
- [ ] Integration with driving simulators

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM fine-tuning
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for the vision-language model
- [NVIDIA](https://huggingface.co/nvidia) for the PhysicalAI dataset
- [AMD ROCm](https://rocm.docs.amd.com/) for GPU compute support

---

<div align="center">

**Made with ❤️ for the AMD GPU Pervasive AI Challenge**

</div>