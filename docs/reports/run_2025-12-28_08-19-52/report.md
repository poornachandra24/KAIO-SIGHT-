# 📑 Training Report — 2025-12-28_08-19-52

**Status:** 🔄 RUNNING  
**Project:** `docs`  

## 📊 Executive Summary
**Duration:** 18.82 hours  
**Steps:** 1925  
**Trainable Params:** 10,092,544 (0.12%)  
**Total Samples**:0

| Metric | Value |
|--------|-------|
| Samples Processed | 30,800 |
| **Initial Loss** | **0.1172** |
| **Final Loss** | **0.0984** |
| Peak VRAM | 17.97 GB |
| Avg VRAM | 17.97 GB |
| Avg GPU Utilization | 49.5 % |
| Avg Temp | 45.0 °C |
| **Peak Power** | **466.0 W** |
| Avg Power | 449.0 W |
| **Total Energy** | **8.4519 kWh** |

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
    "loss": 0.0925,
    "grad_norm": 0.43069788813591003,
    "learning_rate": 1.7827012508505047e-06,
    "epoch": 0.9240048250904704,
    "step": 1915,
    "timestamp": "2025-12-29T03:00:21.323831",
    "vram_gb": 17.971139907836914,
    "power_avg": 449.4,
    "power_min": 439.0,
    "power_max": 455.0,
    "temp": 44.0,
    "util": 49.8
  },
  {
    "loss": 0.097,
    "grad_norm": 0.5652773976325989,
    "learning_rate": 1.672964266871313e-06,
    "epoch": 0.9264173703256936,
    "step": 1920,
    "timestamp": "2025-12-29T03:04:46.883973",
    "vram_gb": 17.971139907836914,
    "power_avg": 452.8,
    "power_min": 450.0,
    "power_max": 455.0,
    "temp": 45.0,
    "util": 50.4
  },
  {
    "loss": 0.0984,
    "grad_norm": 0.5173410177230835,
    "learning_rate": 1.5666555106875425e-06,
    "epoch": 0.9288299155609168,
    "step": 1925,
    "timestamp": "2025-12-29T03:09:12.715815",
    "vram_gb": 17.971139907836914,
    "power_avg": 452.6,
    "power_min": 449.0,
    "power_max": 458.0,
    "temp": 46.0,
    "util": 50.2
  }
]
```