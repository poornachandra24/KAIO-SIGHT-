# 🏗️ System Architecture

## High-Level System Design
The KAIO-SIGHT system is designed as a high-throughput factory for converting raw autonomous vehicle video into reasoning capabilities.

```mermaid
graph TD
    subgraph Data_Plane [Stage 1: Data Setup]
        A[NVIDIA HF Repo] -->|Download| B[Raw Zips]
        B -->|Extract| C[MP4 Video]
        C -->|OpenCV Decode| D[Frames]
        D -->|Resize 504x504| E[Tensors]
        E -->|Map-Style Save| F[Binary Arrow Shards]
    end

    subgraph Compute_Plane [Stage 2: MI300X Training]
        F -->|Lazy Load| G[RAM Cache]
        G -->|Dynamic Batching| H[Unsloth Trainer]
        H -->|Native BF16| I[Qwen2.5-VL Base]
        I -->|LoRA Updates| J[Fine-Tuned Adapters]
    end

    subgraph Inference_Plane [Stage 3: Inference]
        K[User Video] -->|Pre-process| L[Vision Encoder]
        L -->|Features| M[LLM Decoder]
        J -->|Inject| M
        M -->|Generate| N[Driving Decision]
    end
```

## Model Architecture
*   **Backbone:** Qwen2.5-VL-7B-Instruct
*   **Vision Tower:** Internally handles dynamic resolution (up to 4096px), clamped to 504px for stability.
*   **Adapter:** Low-Rank Adaptation (LoRA) targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`.
