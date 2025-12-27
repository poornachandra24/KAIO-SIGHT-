# Inference Flow

## Overview
The inference stage demonstrates how the trained model (`Thunderbird2410/AETHER-THINK-OMNI-Qwen2.5-VL-LoRA`) processes new video data to predict ego-motion.

## The Script (`src/inference/test_inference.py`)
This script loads the fine-tuned model and runs inference on a sample video.

### Workflow
1.  **Model Loading**:
    - Uses `unsloth.FastVisionModel` for optimized loading.
    - Loads the model in `bfloat16` (native precision for MI300X).
    - Enables `FastVisionModel.for_inference(model)` for 2x speedup.
2.  **Input Processing**:
    - Reads a video file using `cv2`.
    - Extracts 16 frames uniformly.
    - Converts frames to RGB.
3.  **Prompt Construction**:
    - Creates a chat template message with the video frames and the instruction: `"Analyze the 4-cam sequence. Predict ego-motion."`.
4.  **Generation**:
    - Tokenizes the input and moves tensors to CUDA (MI300X).
    - Generates the response using `model.generate`.
5.  **Output**:
    - Decodes and prints the predicted action (displacement/velocity).

## Data Lifecycle: Inference Stage
1.  **Input**: Raw MP4 video file.
2.  **Preprocessing**: On-the-fly frame extraction and resizing (in memory).
3.  **Model**: The model processes the visual tokens and text instruction.
4.  **Output**: Text prediction of the vehicle's motion.
