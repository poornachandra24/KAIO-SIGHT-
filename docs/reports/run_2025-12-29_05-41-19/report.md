# 📑 Training Report — 2025-12-29_05-41-19

**Status:** ✅ COMPLETED  
**Project:** `docs`  

## 📊 Executive Summary
**Duration:** 1.81 hours  
**Steps:** 2070  
**Trainable Params:** 10,092,544 (0.12%)  
**Total Samples**:0

| Metric | Value |
|--------|-------|
| Samples Processed | 33,120 |
| **Initial Loss** | **0.0904** |
| **Final Loss** | **0.0948** |
| Peak VRAM | 17.97 GB |
| Avg VRAM | 17.97 GB |
| Avg GPU Utilization | 49.6 % |
| Avg Temp | 45.1 °C |
| **Peak Power** | **465.0 W** |
| Avg Power | 450.4 W |
| **Total Energy** | **0.8162 kWh** |

---

## 🖥️ MI300X Hardware Telemetry
> **Power Chart:** The shaded red region indicates the Min/Max fluctuation per step.

![Telemetry](./telemetry.png)

## ⚙️ Configuration Snapshot

| Hyperparameter | Value |
|----------------|-------|
| **Base Model** | `Qwen2_5_VLForConditionalGeneration` |
| **Precision** | `Bfloat16` |
| **Batch Size (Per GPU)** | `8` |
| **Grad Accumulation** | `2` |
| **Effective Batch Size** | `16` |
| **Learning Rate** | `0.0001` |
| **Optimizer** | `OptimizerNames.PAGED_ADAMW_8BIT` |

---

## 📝 Latest Logs
```json
[
  {
    "loss": 0.091,
    "grad_norm": 0.4613019824028015,
    "learning_rate": 5.745909513127368e-09,
    "epoch": 0.9963811821471653,
    "step": 2065,
    "timestamp": "2025-12-29T07:23:43.211943",
    "vram_gb": 17.971139907836914,
    "power_avg": 448.6,
    "power_min": 438.0,
    "power_max": 455.0,
    "temp": 45.0,
    "util": 49.2
  },
  {
    "loss": 0.0948,
    "grad_norm": 0.5615072846412659,
    "learning_rate": 1.1350119167807817e-09,
    "epoch": 0.9987937273823885,
    "step": 2070,
    "timestamp": "2025-12-29T07:28:05.844011",
    "vram_gb": 17.971139907836914,
    "power_avg": 453.6,
    "power_min": 449.0,
    "power_max": 460.0,
    "temp": 45.0,
    "util": 50.4
  },
  {
    "train_runtime": 6519.9382,
    "train_samples_per_second": 5.085,
    "train_steps_per_second": 0.318,
    "total_flos": 1.6165498957568084e+19,
    "train_loss": 0.0064366136952062995,
    "epoch": 1.0,
    "step": 2073,
    "timestamp": "2025-12-29T07:30:01.284114",
    "vram_gb": 16.59017848968506,
    "power_avg": 443.0,
    "power_min": 420.0,
    "power_max": 458.0,
    "temp": 45.0,
    "util": 49.0
  }
]
```

## 🚀 HuggingFace Deployment
**Repo:** [Thunderbird2410/KAIO-SIGHT](https://huggingface.co/Thunderbird2410/KAIO-SIGHT)
**Commit:** `3cb38077febd83110e8a3b87cb00bc9fce8d631a`
**Adapter Config:** [adapter_config.json](https://huggingface.co/Thunderbird2410/KAIO-SIGHT/blob/3cb38077febd83110e8a3b87cb00bc9fce8d631a/adapter_config.json)
