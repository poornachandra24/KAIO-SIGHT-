# 🪵 Architectural Decision Log (ADR)

| Challenge | Root Cause | Architectural Solution |
| :--- | :--- | :--- |
| **`BrokenProcessPool` Error** | **OpenCV is not fork-safe.** Python's `multiprocessing` pool created corrupted thread states, causing silent segfaults in worker processes. | **Pivot to OS-Level Parallelism & `spawn` Context.** Re-architected `prepare_dataset.py` to be a single-shard worker launched by a Bash script, and forced the multiprocessing context to `spawn` for clean memory isolation. |
| **Extreme Slowness (~7s/sample)** | **I/O Starvation ("Triple Tax").** 1) Re-extracting MP4s from ZIPs for every sample. 2) Slow `cv2.seek()` on compressed video. 3) Multiple processes fighting over the same file handles. | **Implement a Decoupled, Clip-Centric ETL Pipeline.** The `prepare_dataset.py` script now processes data clip-by-clip, reading each video linearly once, which is 10-20x faster. |
| **High RAM Usage during Training** | **Loading hundreds of GBs of processed data.** A naive approach of loading all shards into memory would exhaust the 240GB of system RAM. | **Leverage Hugging Face `datasets` Memory-Mapping.** Instead of a custom `IterableDataset`, the final architecture uses `concatenate_datasets`. This creates a unified "Map-Style" dataset that uses `mmap` to access data directly from the NVMe disk, keeping active RAM usage low. |
| **Tensor Shape Mismatch** | **Collator/Model Disagreement.** The `SFTTrainer` was not correctly interpreting the list of 16 video frames. | **Implement the `UnslothVisionDataCollator`** and refactor the formatting function to separate the text prompt from the raw image list, allowing the collator to handle the tensorization correctly. |
| **Broadcasting Crash** | Training with `Batch Size=2` on variable length visual sequences caused positional embedding mismatches. | **Enforced BS=1, GradAccum=8.** Maintains effective batch size of 8 while ensuring tensor stability. |
| **Empty Tensor Crash** | Image Processor downsampled video frames to 1x1 pixels due to metadata mismatches. | **Clamped to 504x504.** Ensured valid grid formation. |

## Strategic Decision: 4-Camera "Cruciform" Baseline

A key strategic decision was to begin with a 4-camera "Cruciform" setup rather than immediately using all 7 available cameras. This is a standard practice in production-level AI for managing complexity and ensuring a stable baseline.

### Architect's Justification

1.  **De-risking VRAM and Context Length**: A 7-camera setup feeds 112 images (16 frames * 7 cameras) into the model per sample. This creates immense pressure on both the 192GB VRAM and the model's attention mechanism ("token budget"). Starting with a 4-camera setup (64 images) provides a stable, high-performance baseline without risking Out-of-Memory errors or "attention dilution."
2.  **Faster Iteration & Debugging**: The ETL process for 4 cameras is nearly twice as fast as for 7. This allows for rapid iteration during the critical initial phase of model development. If the model fails to converge, it is easier to debug the learning dynamics with a simpler, yet still comprehensive, world-view.
3.  **Designed for Scalability**: The entire system is architected to be modular. To scale up to the full 7-camera view, only a single line in `configs/finetuning_config.yaml` needs to be changed (`camera_setup: "7-cam"`). The `audit.py`, `prepare_dataset.py`, and `trainer.py` scripts will automatically adapt to the new configuration. This demonstrates a "design for scale" philosophy.
