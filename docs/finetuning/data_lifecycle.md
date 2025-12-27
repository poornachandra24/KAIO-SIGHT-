# Data Lifecycle: Finetuning Stage

## Flow
1.  **Input**: Binary Shards (Hugging Face Dataset format) located in `data/processed_dataset`.
2.  **Loading**:
    - The `ShardedDataset` identifies all shard files.
    - It assigns specific shards to each DataLoader worker.
3.  **Streaming**:
    - Workers read samples (Tiled Images + Text Instructions) from disk.
    - Data is streamed into the model batch by batch.
4.  **Training**:
    - The model (e.g., Qwen-VL) processes the tiled images and instructions.
    - It predicts the "Action" (displacement/velocity).
    - Loss is calculated and backpropagated.

## Optimization
- **Pre-computed Tensors**: Because images are resized and tiled during the ETL stage, the training loop does minimal CPU work.
- **RAM Management**: The streaming approach ensures that RAM usage remains constant regardless of dataset size.
