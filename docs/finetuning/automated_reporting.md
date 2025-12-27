# 📊 Automated Reporting

## Overview
The **KAIO-SIGHT** pipeline includes a built-in `AutomatedReportCallback` that generates detailed training reports and telemetry visualizations automatically. This ensures you have a permanent record of every training run's performance and hardware utilization.

## Report Location
Reports are generated in the `docs/reports` directory, organized by timestamp:

```text
docs/reports/
└── run_YYYY-MM-DD_HH-MM-SS/
    ├── report.md       # Executive summary and metrics
    └── telemetry.png   # 6-panel hardware telemetry chart
```

## Report Contents (`report.md`)

The markdown report provides a high-level summary of the training session:

### 1. Executive Summary
-   **Status**: ✅ COMPLETED or 🔄 RUNNING
-   **Duration**: Total training time in hours.
-   **Steps**: Total optimization steps performed.
-   **Trainable Params**: Number and percentage of parameters updated.

### 2. Key Metrics Table
A quick-glance table containing:
-   **Loss**: Initial vs. Final loss.
-   **VRAM**: Peak and Average memory usage.
-   **Power**: Peak and Average power draw (Watts).
-   **Energy**: Total energy consumed (kWh).
-   **Throughput**: Total samples processed.

### 3. Configuration Snapshot
Records the exact hyperparameters used for the run, including:
-   Base Model Architecture
-   Precision (Bfloat16/Float32)
-   Batch Size & Gradient Accumulation
-   Learning Rate & Optimizer

## Telemetry Visualization (`telemetry.png`)

The callback generates a 6-panel chart to visualize system performance over time:

1.  **Training Loss**: Raw (gray) and Smoothed (orange) loss curves.
2.  **Learning Rate**: Scheduler progression.
3.  **VRAM Usage**: Memory footprint on the primary GPU.
4.  **GPU Utilization**: Compute utilization percentage.
5.  **Power Draw**: Real-time power consumption (with min/max shading).
6.  **Temperature**: Peak junction temperature.

## How to Use
This feature is enabled by default in `src/training/trainer.py`. No additional configuration is required.

To view a report, simply navigate to the generated directory and open `report.md` or `telemetry.png`.
