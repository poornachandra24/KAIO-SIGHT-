import os
import yaml
import shutil
import concurrent.futures
import numpy as np
import zipfile
import io
import pandas as pd
import json
from PIL import Image
from datasets import Dataset as HFDataset, concatenate_datasets, load_from_disk
import multiprocessing as mp
import traceback
import torch
import torch.nn.functional as F

# --- GPU-ACCELERATED WORKER FUNCTION (ROCm Compatible) ---
def process_micro_batch(worker_id, chunk_id, clip_uuids, data_cfg_path, ft_cfg_path, shard_path):
    """
    GPU-Accelerated Worker with Arrow overflow fix.
    Compatible with AMD ROCm via PyTorch 'cuda' (HIP) APIs.
    """
    try:
        import sys
        import time
        import psutil
        import gc
        from collections import deque
        import cv2
        
        # Helper for immediate logging
        def log(msg):
            print(f"[GPU-Worker {worker_id}] {msg}", flush=True)

        # Initialize GPU (ROCm/HIP)
        if torch.cuda.is_available():
            device_id = worker_id % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
            log(f"Initialized on ROCm GPU: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device("cpu")
            log("WARNING: No ROCm GPU detected, falling back to CPU")

        cv2.setNumThreads(0)

        # Check if already done
        if os.path.exists(os.path.join(shard_path, "dataset_info.json")):
            return f"Batch {os.path.basename(shard_path)} exists."

        # Load Configs
        with open(data_cfg_path, 'r') as f: d_cfg = yaml.safe_load(f)
        with open(ft_cfg_path, 'r') as f: f_cfg = yaml.safe_load(f)
        
        local_dir = d_cfg['dataset']['local_dir']
        setup_key = f_cfg['vision']['camera_setup']
        camera_names = f_cfg['vision']['setups'][setup_key]['cameras']
        grid = f_cfg['vision']['setups'][setup_key]['grid']
        window_size = f_cfg['vision']['window_size']
        stride = max(1, window_size // 2)
        cols, rows = grid

        # Worker Isolation - save directly to persistent storage for checkpointing
        pid = os.getpid()
        tmp_base = f"/dev/shm/proc_{worker_id}_{pid}"  # For MP4 extraction only
        os.makedirs(tmp_base, exist_ok=True)
        os.makedirs(shard_path, exist_ok=True)  # Create shard dir early for checkpointing
        
        micro_shards = []
        micro_shard_idx = 0
        current_samples = []
        FLUSH_THRESHOLD = 30  # Reduced to avoid Arrow issues

        def flush_to_disk():
            nonlocal micro_shard_idx, current_samples
            if not current_samples: return
            # CHECKPOINT FIX: Save directly to persistent shard_path, not /dev/shm
            save_path = os.path.join(shard_path, f"part_{micro_shard_idx}")
            HFDataset.from_list(current_samples).save_to_disk(save_path)
            micro_shards.append(save_path)
            micro_shard_idx += 1
            current_samples = []
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log(f"Checkpointed micro-shard {micro_shard_idx} to disk")

        chunk_str = f"{chunk_id:04d}"
        
        # Open Zips
        cam_zips = {cam: zipfile.ZipFile(os.path.join(local_dir, "camera", cam, f"{cam}.chunk_{chunk_str}.zip"), 'r') for cam in camera_names}
        ego_zip = zipfile.ZipFile(os.path.join(local_dir, "labels", "egomotion", f"egomotion.chunk_{chunk_str}.zip"), 'r')

        for i, uuid in enumerate(clip_uuids):
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            log(f"Processing clip {i+1}/{len(clip_uuids)}: {uuid} (Mem: {mem_mb:.2f} MB)")
            
            try:
                # 1. Extract MP4s
                caps = {}
                valid_clip = True
                for cam in camera_names:
                    mp4_name = f"{uuid}.{cam}.mp4"
                    target_path = os.path.join(tmp_base, mp4_name)
                    if mp4_name in cam_zips[cam].namelist():
                        cam_zips[cam].extract(mp4_name, tmp_base)
                        caps[cam] = cv2.VideoCapture(target_path)
                    else:
                        valid_clip = False
                        break
                
                if not valid_clip: 
                    continue

                # 2. Load Kinematics
                p_file = f"{uuid}.egomotion.parquet"
                with ego_zip.open(p_file) as f:
                    kin_df = pd.read_parquet(io.BytesIO(f.read()))
                    time_col = next((c for c in ['timestamp_us', 'timestamp_ns', 'timestamp'] if c in kin_df.columns), kin_df.columns[0])
                    kin_df = kin_df.sort_values(time_col)
                    kinematics = kin_df[['x', 'y', 'vx', 'vy']].values.tolist()

                # 3. GPU-Accelerated Rolling Buffer
                frame_buffer = deque(maxlen=window_size)
                
                for frame_idx in range(len(kinematics)):
                    current_frames = {}
                    
                    for cam in camera_names:
                        ret, frame = caps[cam].read()
                        if ret:
                            # GPU ACCELERATION: Upload and resize on GPU
                            # Frame is BGR HWC uint8
                            tensor = torch.from_numpy(frame).to(device)  # HWC
                            tensor = tensor.permute(2, 0, 1).float()  # CHW
                            # BGR to RGB
                            tensor = tensor[[2, 1, 0], :, :]
                            # Resize on GPU
                            tensor = F.interpolate(
                                tensor.unsqueeze(0), 
                                size=(896, 896), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                            current_frames[cam] = tensor
                        else:
                            current_frames[cam] = torch.zeros((3, 896, 896), device=device)
                    
                    frame_buffer.append(current_frames)

                    window_start_idx = frame_idx - window_size + 1
                    
                    if len(frame_buffer) == window_size and window_start_idx >= 0 and window_start_idx % stride == 0:
                        # GPU TILING
                        window_imgs = []
                        w, h = 896, 896
                        
                        for t in range(window_size):
                            # Create canvas on GPU
                            canvas = torch.zeros((3, h * rows, w * cols), device=device)
                            frames_at_t = frame_buffer[t]
                            
                            for idx, cam in enumerate(camera_names):
                                if idx >= cols * rows: 
                                    break
                                c_x, c_y = idx % cols, idx // cols
                                canvas[:, c_y*h:(c_y+1)*h, c_x*w:(c_x+1)*w] = frames_at_t[cam]
                            
                            # Convert to PIL for saving (move to CPU)
                            canvas_cpu = canvas.clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
                            window_imgs.append(Image.fromarray(canvas_cpu))

                        # Physics
                        p_start = np.array(kinematics[window_start_idx][:2])
                        p_end = np.array(kinematics[frame_idx][:2])
                        dist = np.linalg.norm(p_end - p_start)
                        v_avg = np.mean([np.linalg.norm([k[2], k[3]]) for k in kinematics[window_start_idx:frame_idx+1]])

                        current_samples.append({
                            "images": window_imgs,
                            "instruction": f"Analyze the {setup_key} sequence. Predict ego-motion.",
                            "reasoning": f"Clip {uuid}: Velocity {v_avg:.1f}m/s. Displacement {dist:.2f}m.",
                            "action": f"Displacement: {dist:.2f}m"
                        })
                        
                        if len(current_samples) >= FLUSH_THRESHOLD:
                            flush_to_disk()

                # Cleanup video handles and files
                for c in caps.values(): 
                    c.release()
                for f in os.listdir(tmp_base): 
                    if f.endswith(".mp4"): 
                        os.remove(os.path.join(tmp_base, f))

            except Exception as e:
                log(f"Error processing clip {uuid}: {e}")
                traceback.print_exc()

        # Final flush
        flush_to_disk()
        
        for z in cam_zips.values(): 
            z.close()
        ego_zip.close()
        
        # ARROW OVERFLOW FIX: Don't merge, just move micro-shards
        if micro_shards:
            os.makedirs(shard_path, exist_ok=True)
            total_samples = 0
            
            for i, ms_path in enumerate(micro_shards):
                dest = os.path.join(shard_path, f"part_{i}")
                shutil.move(ms_path, dest)
                try:
                    ds = load_from_disk(dest)
                    total_samples += len(ds)
                except: 
                    pass
            
            # Create marker file
            with open(os.path.join(shard_path, "dataset_info.json"), "w") as f:
                json.dump({"num_parts": len(micro_shards), "total_samples": total_samples}, f)
            
            log(f"Saved {total_samples} samples in {len(micro_shards)} parts")
            
            try: 
                shutil.rmtree(tmp_base) 
            except: 
                pass
            
            return f"Batch {os.path.basename(shard_path)}: {total_samples} samples."
        
        try: 
            shutil.rmtree(tmp_base) 
        except: 
            pass
        
        log("No samples generated.")
        return f"Batch {os.path.basename(shard_path)}: Empty."

    except Exception as e:
        return f"CRASH in Worker {worker_id}: {str(e)}\n{traceback.format_exc()}"


# --- MAIN ORCHESTRATOR ---
def prepare():
    from src.data_etl.loader import NuRecTemporalDataset
    
    PROJECT_ROOT = "/workspace/AMD-Vision-Omni"
    data_cfg = f"{PROJECT_ROOT}/configs/data_config.yaml"
    ft_cfg = f"{PROJECT_ROOT}/configs/finetuning_config.yaml"
    SHARD_DIR = f"{PROJECT_ROOT}/data/shards"
    FINAL_OUTPUT = f"{PROJECT_ROOT}/data/processed_dataset"
    
    # GPU workers - use fewer to avoid VRAM contention
    NUM_WORKERS = 8
    CLIPS_PER_BATCH = 5

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: 
        pass

    os.makedirs(SHARD_DIR, exist_ok=True)
    print(f"--- [TURBO ETL: GPU ACCELERATED (ROCm)] ---")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("Scanning chunks...")
    raw_ds = NuRecTemporalDataset(data_cfg, ft_cfg, split='train')
    
    chunk_map = {}
    for s in raw_ds.samples:
        c_id = s['chunk']
        u_id = s['clip_uuid']
        if c_id not in chunk_map: 
            chunk_map[c_id] = set()
        chunk_map[c_id].add(u_id)
    
    tasks = []
    batch_idx = 0
    
    for chunk_id, uuids in chunk_map.items():
        uuid_list = list(uuids)
        for i in range(0, len(uuid_list), CLIPS_PER_BATCH):
            batch_uuids = uuid_list[i : i + CLIPS_PER_BATCH]
            shard_path = os.path.join(SHARD_DIR, f"batch_{batch_idx}")
            tasks.append((chunk_id, batch_uuids, shard_path))
            batch_idx += 1

    print(f"Created {len(tasks)} micro-tasks.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, (cid, u_list, s_path) in enumerate(tasks):
            futures.append(executor.submit(
                process_micro_batch, i % NUM_WORKERS, cid, u_list, data_cfg, ft_cfg, s_path
            ))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if "CRASH" in result:
                print(f"❌ {result}")
            else:
                print(f"✔️ {result}")

    # Final consolidation - load all parts
    print("\n--- [CONSOLIDATING] ---")
    all_parts = []
    
    for i in range(batch_idx):
        batch_path = os.path.join(SHARD_DIR, f"batch_{i}")
        if os.path.exists(os.path.join(batch_path, "dataset_info.json")):
            # Load all parts from this batch
            for part_dir in sorted(os.listdir(batch_path)):
                if part_dir.startswith("part_"):
                    part_path = os.path.join(batch_path, part_dir)
                    try:
                        all_parts.append(load_from_disk(part_path))
                    except Exception as e:
                        print(f"Warning: Could not load {part_path}: {e}")
    
    if all_parts:
        print(f"Merging {len(all_parts)} parts (in batches to avoid overflow)...")
        
        # Merge in batches of 10 to avoid Arrow overflow
        MERGE_BATCH = 10
        merged_datasets = []
        
        for i in range(0, len(all_parts), MERGE_BATCH):
            batch = all_parts[i:i+MERGE_BATCH]
            merged = concatenate_datasets(batch)
            merged_datasets.append(merged)
            print(f"  Merged parts {i} to {i+len(batch)-1}")
        
        # Final merge
        if len(merged_datasets) > 1:
            final_ds = concatenate_datasets(merged_datasets)
        else:
            final_ds = merged_datasets[0]
        
        final_ds.save_to_disk(FINAL_OUTPUT)
        print(f"✅ ETL COMPLETE: {len(final_ds)} samples saved to {FINAL_OUTPUT}")
        
        # Cleanup shards
        shutil.rmtree(SHARD_DIR)
    else:
        print("❌ No data processed.")


if __name__ == "__main__":
    prepare()