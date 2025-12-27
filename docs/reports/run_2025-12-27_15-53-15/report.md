# 📑 Training Report — 2025-12-27_15-53-15

**Status:** 🔄 RUNNING  
**Project:** `docs`  

## 📊 Executive Summary
**Duration:** 0.08 hours  
**Steps:** 25  
**Trainable Params:** 10,092,544 (0.12%)  
**Total Examples:** 0  

| Metric | Value |
|--------|-------|
| Samples Processed | 200 |
| **Initial Loss** | **5.9054** |
| **Final Loss** | **5.4721** |
| Peak VRAM | 16.14 GB |
| Avg VRAM | 16.14 GB |
| Avg GPU Utilization | 50.1 % |
| Avg Temp | 44.6 °C |
| **Peak Power** | **474.0 W** |
| Avg Power | 463.4 W |
| **Total Energy** | **0.0383 kWh** |

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
    "loss": 5.8825,
    "grad_norm": 13.046425819396973,
    "learning_rate": 3.3734939759036146e-06,
    "epoch": 0.003619581938286128,
    "step": 15,
    "timestamp": "2025-12-27T15:56:25.155694",
    "vram_gb": 16.139089584350586,
    "power_avg": 465.2,
    "power_min": 461.0,
    "power_max": 468.0,
    "temp": 45.0,
    "util": 50.2
  },
  {
    "loss": 5.7136,
    "grad_norm": 13.449252128601074,
    "learning_rate": 4.578313253012049e-06,
    "epoch": 0.00482610925104817,
    "step": 20,
    "timestamp": "2025-12-27T15:57:17.939681",
    "vram_gb": 16.137075424194336,
    "power_avg": 466.2,
    "power_min": 462.0,
    "power_max": 469.0,
    "temp": 45.0,
    "util": 50.4
  },
  {
    "loss": 5.4721,
    "grad_norm": 12.17174243927002,
    "learning_rate": 5.783132530120483e-06,
    "epoch": 0.006032636563810213,
    "step": 25,
    "timestamp": "2025-12-27T15:58:12.307695",
    "vram_gb": 16.139089584350586,
    "power_avg": 465.0,
    "power_min": 457.0,
    "power_max": 474.0,
    "temp": 45.0,
    "util": 50.2
  }
]
```