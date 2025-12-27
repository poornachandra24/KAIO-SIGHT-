# 🛡️ Challenges & Resolutions

### 1. The "Broadcasting" Crash
*   **Error:** `RuntimeError: The size of tensor a (2) must match the size of tensor b (1600)`
*   **Cause:** Training with `Batch Size=2` on variable length visual sequences. The positional embeddings (1600 patches) could not broadcast across two different grid sizes.
*   **Fix:** Forced `per_device_train_batch_size = 1` and increased `gradient_accumulation_steps`.

### 2. The "Empty Tensor" Crash
*   **Error:** `RuntimeError: shape '[0, 4, -1]' is invalid for input of size 1280`
*   **Cause:** The Image Processor, without explicit constraints, downsampled video frames to a single token (14x14px) due to metadata mismatches, causing the Spatial Merge layer to fail.
*   **Fix:** Hard-coded `tokenizer.image_processor.min_pixels = 504*504` and ensured manual resizing in the transform function.

### 3. The "Missing Config" Inference Failure
*   **Error:** `RuntimeError: Unsloth: No config file found`
*   **Cause:** Unsloth's merged push mechanism uploaded the weights (safetensors) but failed to copy `config.json` from the base model.
*   **Fix:** Created `scripts/fix_repo.py` to pull config files from the base Qwen repo and patch the fine-tuned repo.
