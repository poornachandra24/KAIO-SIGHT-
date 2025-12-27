# 📑 Training Report — 2025-12-27_13-39-42

**Status:** ✅ COMPLETED  
**Project:** `docs`  

## 📊 Executive Summary
**Duration:** 0.07 hours  
**Steps:** 20  
**Trainable Params:** 10,092,544 (0.12%)  
**Total Examples:** 0  

| Metric | Value |
|--------|-------|
| Samples Processed | 160 |
| **Initial Loss** | **5.1559** |
| **Final Loss** | **1.8635** |
| Peak VRAM | 16.14 GB |
| Avg VRAM | 16.14 GB |
| Avg GPU Utilization | 49.9 % |
| Avg Temp | 44.8 °C |
| **Peak Power** | **469.0 W** |
| Avg Power | 462.8 W |
| **Total Energy** | **0.0324 kWh** |

---

## 🖥️ MI300X Hardware Telemetry
> **Power Chart:** The shaded red region indicates the Min/Max fluctuation per step.

![Telemetry](./telemetry.png)

## ⚙️ Configuration Snapshot

| Hyperparameter | Value |
|----------------|-------|
| **Base Model** | `Qwen2_5_VLForConditionalGeneration` |
| **Precision** | `Bfloat16` |
| **Batch Size (Per GPU)** | `1` |
| **Grad Accumulation** | `8` |
| **Effective Batch Size** | `8` |
| **Learning Rate** | `0.0001` |
| **Optimizer** | `OptimizerNames.PAGED_ADAMW_8BIT` |

---

## 📝 Latest Logs
```json
[
  {
    "loss": 2.205,
    "grad_norm": 4.4009599685668945,
    "learning_rate": 3.3333333333333335e-05,
    "epoch": 0.003619581938286128,
    "step": 15,
    "timestamp": "2025-12-27T13:42:57.691655",
    "vram_gb": 16.139089584350586,
    "power_avg": 463.8,
    "power_min": 457.0,
    "power_max": 468.0,
    "temp": 45.0,
    "util": 49.8
  },
  {
    "loss": 1.8635,
    "grad_norm": 4.24073600769043,
    "learning_rate": 5.555555555555556e-06,
    "epoch": 0.00482610925104817,
    "step": 20,
    "timestamp": "2025-12-27T13:43:53.019655",
    "vram_gb": 16.137075424194336,
    "power_avg": 465.8,
    "power_min": 461.0,
    "power_max": 469.0,
    "temp": 45.0,
    "util": 50.2
  },
  {
    "train_runtime": 248.9902,
    "train_samples_per_second": 0.643,
    "train_steps_per_second": 0.08,
    "total_flos": 2.008997387512627e+16,
    "train_loss": 3.0488024234771727,
    "epoch": 0.00482610925104817,
    "step": 20,
    "timestamp": "2025-12-27T13:43:53.637340",
    "vram_gb": 16.137075424194336,
    "power_avg": 0,
    "power_min": 0,
    "power_max": 0,
    "temp": 0,
    "util": 0
  }
]
```