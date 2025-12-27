# 🔄 Data Lifecycle

## 1. Ingestion (Raw)
*   **Format:** Zip archives (`camera_front.zip`, `egomotion.parquet`).
*   **Storage:** `/data/raw`.
*   **Audit:** Checksums verified against NVIDIA manifest.

### Architect's Insight: `target_chunks` vs. Sample Count
Setting `target_chunks: 1` can result in over 33,000 training samples. This reveals the "nested" nature of the NVIDIA NuRec dataset.
-   **`target_chunks: 1`** instructs the `downloader.py` to download a single "chunk" file.
-   However, this one `.zip` file contains **hundreds of individual `.mp4` video clips**.
-   **The Math:** `1 Chunk` $\times$ `~100 Clips/Chunk` $\times$ `~28 Samples/Clip` = **Thousands of training samples.**

## 2. ETL Transformation
*   **Trigger:** `src/data_etl/prepare_dataset.py`.
*   **Process:**
    1.  Unzip on-the-fly.
    2.  Extract 16 frames @ 10Hz.
    3.  Sync with Egomotion (dx, dy, yaw).
    4.  Generate Instruction: "Analyze 4-cam sequence..."
    5.  Generate Output: `<think>...</think> Action: ...`
*   **Output:** Binary Arrow Shards (`/data/shards/batch_X`).

## 3. Training Load
*   **Mechanism:** **Map-Style Dataset** via `datasets.load_from_disk`.
*   **Zero-Copy Loading:** Uses OS-level **Memory Mapping (`mmap`)**. The binary shards on disk are mapped directly into virtual memory, allowing the trainer to access 100GB+ datasets with near-zero RAM overhead.
*   **Consolidation:** Individual shards are stitched together using `concatenate_datasets` into a single virtual dataset.
*   **Transform:** On-the-fly Resize to **504x504**.
*   **Tensorization:** `UnslothVisionDataCollator` pads text and flattens images to 1D vectors.

## 4. Inference
*   **Input:** MP4 Video.
*   **Process:** OpenCV Decode -> Resize -> Tokenize -> Generate.
