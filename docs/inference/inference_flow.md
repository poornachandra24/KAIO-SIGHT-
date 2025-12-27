# Inference Flow

## Overview
The inference stage demonstrates how the trained model (`Thunderbird2410/KAIO-SIGHT`) processes new video data to predict ego-motion.

## The Script (`scripts/03_inference.sh`)
This script wraps the Python inference logic for easy execution.

### Usage
```bash
bash scripts/03_inference.sh
```

## Inference Logic (`src/inference/test_inference.py`)

### Workflow
1.  **Model Loading**:
    - Loads the **Base Model**: `Qwen/Qwen2.5-VL-7B-Instruct`.
    - Loads the **LoRA Adapter**: `Thunderbird2410/KAIO-SIGHT`.
    - Uses `FastVisionModel` with `bfloat16` precision.
2.  **Input Processing**:
    - Reads a video file using `cv2`.
    - Extracts 16 frames uniformly.
    - Converts frames to RGB.
3.  **Prompt Construction**:
    - Creates a chat template message with the video frames and the instruction: `"Analyze the 4-cam sequence. Predict ego-motion."`.
4.  **Generation**:
    - Tokenizes the input and moves tensors to rocm mapped to cuda (MI300X).
    - Generates the response using `model.generate`.
5.  **Output**:
    - Decodes and prints the predicted action (displacement/velocity).

## Data Lifecycle: Inference Stage
1.  **Input**: Raw MP4 video file.
2.  **Preprocessing**: On-the-fly frame extraction and resizing (in memory).
3.  **Model**: The model processes the visual tokens and text instruction.
4.  **Output**: Text prediction of the vehicle's motion.
