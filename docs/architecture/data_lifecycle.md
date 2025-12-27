# 🔄 Data Lifecycle

## 1. Ingestion (Raw)
*   **Format:** Zip archives (`camera_front.zip`, `egomotion.parquet`).
*   **Storage:** `/data/raw`.
*   **Audit:** Checksums verified against NVIDIA manifest.

## 2. ETL Transformation
*   **Trigger:** `src/data/prepare_dataset.py`.
*   **Process:**
    1.  Unzip on-the-fly.
    2.  Extract 16 frames @ 10Hz.
    3.  Sync with Egomotion (dx, dy, yaw).
    4.  Generate Instruction: "Analyze 4-cam sequence..."
    5.  Generate Output: `<think>...</think> Action: ...`
*   **Output:** Binary Arrow Shards (`/data/shards/batch_X`).

## 3. Training Load
*   **Mechanism:** `datasets.load_from_disk` (Memory Mapped).
*   **Transform:** On-the-fly Resize to **504x504**.
*   **Tensorization:** `UnslothVisionDataCollator` pads text and flattens images to 1D vectors.

## 4. Inference
*   **Input:** MP4 Video.
*   **Process:** OpenCV Decode -> Resize -> Tokenize -> Generate.
