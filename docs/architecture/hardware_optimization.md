# ⚡ System Design & Hardware Optimizations

## Hardware Profile
*   **GPU:** [AMD Instinct MI300X (192GB HBM3)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html).
*   **Stack:** ROCm 6.4, PyTorch 2.9 (HIP), Triton 3.x.

## Optimizations Implemented

### 1. Native Bfloat16
We use 16-bit LoRA (Bfloat16) instead of 4-bit quantization (BNB) to ensure maximum stability and performance on MI300X.
*   **Benefit:** 100% utilization of MI300X matrix cores.
*   **VRAM Usage:** ~17GB / 192GB (leaving massive headroom for context scaling).

### 2. Persistent Triton Cache
Explicitly set `TRITON_CACHE_DIR` to a workspace volume.
*   **Benefit:** Prevents the 3-minute JIT compilation delay on every container restart.

### 3. Zero-Copy Data Loading
Switched from Streaming (Network/Generator) to Map-Style (Disk).
*   **Benefit:** Uses OS-level memory mapping. Linux caches the active shards in RAM automatically, resulting in **Zero-Idle** GPU utilization without manual RAM management.

### 4. File Descriptor tuning
Increased `ulimit -n` to 1,048,576.
*   **Benefit:** Allows opening 1000+ data shards simultaneously without `Too many open files` errors.
