import os
import time
import json
import textwrap
import subprocess
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from datetime import datetime
import torch
import numpy as np

def get_amd_metrics():
    """Scrapes AMD MI300X metrics using rocm-smi."""
    metrics = {'power': 0.0, 'temp': 0.0, 'util': 0.0}
    try:
        # P: Power, t: Temp, u: GPU use
        result = subprocess.check_output(
            ['rocm-smi', '--showpower', '--showtemp', '--showuse', '--csv'], 
            stderr=subprocess.STDOUT
        ).decode('utf-8')
        lines = result.strip().split('\n')
        if len(lines) > 1:
            data = lines[1].split(',')
            for item in data:
                val = item.strip()
                if val.isdigit() and 0 < int(val) <= 100:
                    metrics['util'] = float(val)
                elif '.' in val:
                    f_val = float(val)
                    if f_val > 100: metrics['power'] = f_val
                    elif 20 < f_val < 100: metrics['temp'] = f_val
    except Exception:
        pass 
    return metrics

class AutomatedReportCallback(TrainerCallback):
    def __init__(self, output_dir="docs/reports", num_examples=0):
        self.output_dir = output_dir
        self.history = []
        self.num_examples = num_examples
        self.start_time = time.time()
        self.step_buffer = {'power': [], 'temp': [], 'util': []}
        
        # Model Stats (Captured at start)
        self.trainable_params = 0
        self.total_params = 0
        self.model_arch = "Unknown"

        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        plt.switch_backend("Agg")
        plt.style.use('bmh')
        plt.rcParams.update({
            "figure.figsize": (12, 10),
            "font.size": 10,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.3
        })

    def ema(self, data, alpha=0.1):
        ema_vals = []
        for x in data:
            if not ema_vals: ema_vals.append(x)
            else: ema_vals.append(ema_vals[-1] * (1 - alpha) + x * alpha)
        return ema_vals

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture static model stats once at the start"""
        if model:
            self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.total_params = sum(p.numel() for p in model.parameters())
            self.model_arch = getattr(model.config, "architectures", ["Transformer"])[0]

    def on_step_end(self, args, state, control, **kwargs):
        """Poll Hardware every step"""
        hw = get_amd_metrics()
        self.step_buffer['power'].append(hw['power'])
        self.step_buffer['temp'].append(hw['temp'])
        self.step_buffer['util'].append(hw['util'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        
        entry = logs.copy()
        entry["step"] = state.global_step
        entry["epoch"] = state.epoch
        entry["timestamp"] = datetime.now().isoformat()

        if torch.cuda.is_available():
            # Use max_memory_allocated to capture peak usage during the interval
            entry["vram_gb"] = torch.cuda.max_memory_allocated(0) / 1024**3
            # Reset peak stats for the next interval
            torch.cuda.reset_peak_memory_stats(0)
        
        # Aggregate Hardware Metrics
        if self.step_buffer['power']:
            entry['power_avg'] = np.mean(self.step_buffer['power'])
            entry['power_min'] = np.min(self.step_buffer['power'])
            entry['power_max'] = np.max(self.step_buffer['power'])
            entry['temp'] = np.max(self.step_buffer['temp'])
            entry['util'] = np.mean(self.step_buffer['util'])
        else:
            entry['power_avg'] = 0; entry['power_min'] = 0; entry['power_max'] = 0
            entry['temp'] = 0; entry['util'] = 0

        self.step_buffer = {'power': [], 'temp': [], 'util': []} # Reset
        self.history.append(entry)

        # Generate report every 5 logs
        if len(self.history) % 5 == 0: 
            self.generate_report(args)

    def on_train_end(self, args, state, control, **kwargs):
        self.generate_report(args, final=True)

    def generate_report(self, args, final=False):
        if not self.history: return

        # --- EXTRACT DATA ---
        steps = [x["step"] for x in self.history if "loss" in x]
        if not steps: return
        
        losses = [x["loss"] for x in self.history if "loss" in x]
        lrs = [x.get("learning_rate", 0) for x in self.history if "loss" in x]
        vram = [x.get("vram_gb", 0) for x in self.history if "loss" in x]
        
        # Extract Hardware Data safely (handle missing keys)
        p_avg = [x.get("power_avg", 0) for x in self.history if "loss" in x]
        p_min = [x.get("power_min", 0) for x in self.history if "loss" in x]
        p_max = [x.get("power_max", 0) for x in self.history if "loss" in x]
        temp = [x.get("temp", 0) for x in self.history if "loss" in x]
        util = [x.get("util", 0) for x in self.history if "loss" in x]

        # --- PLOTTING (6-Panel) ---
        fig, axs = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle(f"MI300X Training Telemetry - {self.run_id}", fontsize=16)

        # 1. Loss
        axs[0, 0].plot(steps, losses, alpha=0.3, color="gray", label="Raw")
        axs[0, 0].plot(steps, self.ema(losses), color="#FF5733", label="Smoothed")
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].legend()

        # 2. Learning Rate
        axs[0, 1].plot(steps, lrs, color="#33C1FF")
        axs[0, 1].set_title("Learning Rate")

        # 3. VRAM
        axs[1, 0].plot(steps, vram, color="#33FF57", fillstyle='bottom')
        axs[1, 0].fill_between(steps, vram, color="#33FF57", alpha=0.1)
        axs[1, 0].set_title("VRAM Usage (GB)")

        # 4. Utilization
        axs[1, 1].plot(steps, util, color="#FFC300")
        axs[1, 1].set_title("GPU Compute Utilization (Avg %)")
        axs[1, 1].set_ylim(0, 105)

        # 5. Power
        axs[2, 0].plot(steps, p_avg, color="#E74C3C", label="Average")
        axs[2, 0].fill_between(steps, p_min, p_max, color="#E74C3C", alpha=0.2)
        axs[2, 0].set_title("Power Draw (Watts)")
        axs[2, 0].set_ylabel("Watts")

        # 6. Temp
        axs[2, 1].plot(steps, temp, color="#8E44AD")
        axs[2, 1].set_title("Peak Junction Temp (°C)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.run_dir, "telemetry.png"), dpi=150)
        plt.close()

        # --- METRICS CALCULATION (Explicit Definition) ---
        duration_hrs = (time.time() - self.start_time) / 3600
        
        # Calculate stats safely using numpy
        # 'p_avg' is a list of averages per step-interval. We take the mean of those averages.
        overall_avg_power = np.mean(p_avg) if p_avg else 0.0
        peak_power = np.max(p_max) if p_max else 0.0
        
        avg_vram = np.mean(vram) if vram else 0.0
        avg_util = np.mean(util) if util else 0.0
        avg_temp = np.mean(temp) if temp else 0.0
        
        kwh = (overall_avg_power * duration_hrs) / 1000
        
        status_icon = "✅ COMPLETED" if final else "🔄 RUNNING"
        
        bs = args.per_device_train_batch_size
        ga = args.gradient_accumulation_steps
        eff_bs = bs * ga
        samples_processed = max(steps) * eff_bs
        percent_trained = (self.trainable_params / self.total_params) * 100 if self.total_params > 0 else 0
        
        # --- GENERATE MARKDOWN ---
        md_content = textwrap.dedent(f"""
# 📑 Training Report — {self.run_id}

**Status:** {status_icon}  
**Project:** `{self.output_dir.split('/')[-2]}`  

## 📊 Executive Summary
**Duration:** {duration_hrs:.2f} hours  
**Steps:** {max(steps)}  
**Trainable Params:** {self.trainable_params:,} ({percent_trained:.2f}%)  
**Total Samples**:{self.num_examples}

| Metric | Value |
|--------|-------|
| Samples Processed | {samples_processed:,} |
| **Initial Loss** | **{losses[0]:.4f}** |
| **Final Loss** | **{losses[-1]:.4f}** |
| Peak VRAM | {max(vram):.2f} GB |
| Avg VRAM | {avg_vram:.2f} GB |
| Avg GPU Utilization | {avg_util:.1f} % |
| Avg Temp | {avg_temp:.1f} °C |
| **Peak Power** | **{peak_power:.1f} W** |
| Avg Power | {overall_avg_power:.1f} W |
| **Total Energy** | **{kwh:.4f} kWh** |

---

## 🖥️ MI300X Hardware Telemetry
> **Power Chart:** The shaded red region indicates the Min/Max fluctuation per step.

![Telemetry](./telemetry.png)

## ⚙️ Configuration Snapshot

| Hyperparameter | Value |
|----------------|-------|
| **Base Model** | `{self.model_arch}` |
| **Precision** | `{'Bfloat16' if args.bf16 else 'Float32'}` |
| **Batch Size (Per GPU)** | `{bs}` |
| **Grad Accumulation** | `{ga}` |
| **Effective Batch Size** | `{eff_bs}` |
| **Learning Rate** | `{args.learning_rate}` |
| **Optimizer** | `{args.optim}` |

---

## 📝 Latest Logs
```json
{json.dumps(self.history[-3:], indent=2)}
```
        """).strip()
        with open(os.path.join(self.run_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        if final:
            print(f"📊 Final Report: {self.run_dir}/report.md")