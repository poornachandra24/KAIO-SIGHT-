# Data Lifecycle: Finetuning Stage

## Flow
1.  **Input**: Binary Shards (Hugging Face Dataset format) located in `data/shards`.
2.  **Loading**:
    - Pre-processed data is read directly from disk if target resolution and the pre-processed data's resolution match.
3.  **Streaming**:
    - Workers read samples (Tiled Images + Text Instructions) from disk.
    - Data is streamed into the model batch by batch.
4.  **Training**:
    - The model (e.g., Qwen-VL) processes the tiled images and instructions.
    - It predicts the "Action" (displacement/velocity).
    - Loss is calculated and backpropagated.

## Optimization
- **Pre-computed Tensors**: Because images are resized and tiled during the ETL stage, the training loop does minimal CPU work.
- **RAM Management**: The RAM is utilized if the the target resolution needs to be altered at finetuning stage to meet the limit of the context window of the model
