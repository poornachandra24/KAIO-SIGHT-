# 📊 Automated Reporting

## Overview
The **KAIO-SIGHT** pipeline includes a built-in `AutomatedReportCallback` that generates detailed training reports and telemetry visualizations automatically. This ensures you have a permanent record of every training run's performance and hardware utilization.

## Report Location
Reports are now organized by **Project Name** (defined in `configs/finetuning_config.yaml`) to support resumable training history.

```text
docs/reports/
└── <project_name>/              # e.g., "amd-vision-omni"
    ├── report.md                # Executive summary (updates in-place)
    ├── telemetry.png            # 6-panel hardware telemetry chart
    └── history.json             # Persisted metric history
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
-   **VRAM**: **True Peak** usage (captured via `torch.cuda.max_memory_allocated` per step) and Average.
-   **Power**: Peak and Average power draw (Watts), captured via 2Hz background polling.
-   **Energy**: Total energy consumed (kWh).
-   **Throughput**: Total samples processed.

> [!NOTE]
> Previous versions of this system captured "resting" VRAM/Power at the end of each step. The current implementation uses background polling and peak tracking to capture the genuine hardware load during the forward/backward pass.

### 3. Configuration Snapshot
Records the exact hyperparameters used for the run, including:
-   Base Model Architecture
-   Precision (Bfloat16/Float32)
-   Batch Size & Gradient Accumulation
-   Learning Rate & Optimizer

## Dual-Logging Strategy

### 1. Local Markdown Reports
Generated via `AutomatedReportCallback` in `docs/reports`. Serves as a permanent, offline-accessible record with hardware telemetry.

### 2. Comet ML Integration
This system is natively integrated with **Comet ML** for live, remote monitoring.
-   **Enabled by**: Setting `report_to: ["comet_ml"]` in the trainer config (default).
-   **Features**: Real-time loss curves, system metrics, and hyperparameter tracking.
-   **Setup**: Ensure `COMET_API_KEY` and `COMET_PROJECT_NAME` are set in your environment.

## Telemetry Visualization (`telemetry.png`)

The callback generates a 6-panel chart to visualize system performance over time:

1.  **Training Loss**: Raw (gray) and Smoothed (orange) loss curves.
2.  **Learning Rate**: Scheduler progression.
3.  **VRAM Usage**: Peak memory footprint on the primary GPU.
4.  **GPU Utilization**: Compute utilization percentage.
5.  **Power Draw**: Real-time power consumption (with min/max shading).
6.  **Temperature**: Peak junction temperature.

## How to Use
This feature is enabled by default in `src/training/trainer.py`.

To view a report, navigate to `docs/reports/<project_name>/` and open `report.md` or `telemetry.png`.
