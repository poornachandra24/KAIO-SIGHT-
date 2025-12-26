import os
import yaml
import shutil
import concurrent.futures
import numpy as np
import zipfile
import io
import pandas as pd
from PIL import Image
from datasets import Dataset as HFDataset, concatenate_datasets, load_from_disk
from src.data.loader import NuRecTemporalDataset
import multiprocessing as mp
import traceback

# --- WORKER FUNCTION ---
# --- WORKER FUNCTION ---
# --- WORKER FUNCTION ---
# --- WORKER FUNCTION ---
def process_micro_batch(worker_id, chunk_id, clip_uuids, data_cfg_path, ft_cfg_path, shard_path):
    """
    Worker process with LAZY IMPORTS and AGGRESSIVE MEMORY MANAGEMENT.
    """
    try:
        import sys
        import time
        import psutil
        import gc
        
        # Helper for immediate logging
        def log(msg):
            print(f"[Worker {worker_id}] {msg}", flush=True)

        log(f"Starting batch {os.path.basename(shard_path)} with {len(clip_uuids)} clips.")
        
        # ARCHITECT FIX: Import CV2 here to avoid global lock contention
        import cv2
        from collections import deque
        cv2.setNumThreads(0) # Disable internal threading
        log("CV2 initialized.")

        if os.path.exists(os.path.join(shard_path, "dataset_info.json")):
            log("Shard already exists. Skipping.")
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

        # Worker Isolation
        pid = os.getpid()
        tmp_base = f"/dev/shm/proc_{worker_id}_{pid}"
        os.makedirs(tmp_base, exist_ok=True)
        log(f"Temp dir: {tmp_base}")
        
        # We will save small chunks here and merge them at the end
        micro_shards = []
        micro_shard_idx = 0
        
        # Buffer for current samples
        current_samples = []
        FLUSH_THRESHOLD = 50 # Flush every 50 samples (~7.5GB RAM)

        def flush_to_disk():
            nonlocal micro_shard_idx, current_samples
            if not current_samples: return
            
            save_path = os.path.join(tmp_base, f"micro_{micro_shard_idx}")
            HFDataset.from_list(current_samples).save_to_disk(save_path)
            micro_shards.append(save_path)
            micro_shard_idx += 1
            current_samples = [] # Clear memory
            gc.collect() # Force GC
            log(f"Flushed micro-shard {micro_shard_idx} to disk.")

        chunk_str = f"{chunk_id:04d}"

        # Open Zip Handles
        log("Opening zip files...")
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
                        log(f"Missing {mp4_name} in zip.")
                        valid_clip = False; break
                
                if not valid_clip: continue

                # 2. Load Kinematics
                p_file = f"{uuid}.egomotion.parquet"
                with ego_zip.open(p_file) as f:
                    kin_df = pd.read_parquet(io.BytesIO(f.read()))
                    # Auto-detect timestamp column
                    time_col = next((c for c in ['timestamp_us', 'timestamp_ns', 'timestamp'] if c in kin_df.columns), kin_df.columns[0])
                    kin_df = kin_df.sort_values(time_col)
                    kinematics = kin_df[['x', 'y', 'vx', 'vy']].values.tolist()

                # 3. Rolling Buffer Logic
                frame_buffer = deque(maxlen=window_size)
                
                # Iterate frame by frame
                for frame_idx in range(len(kinematics)):
                    # Read current frame from all cameras
                    current_frames = {}
                    for cam in camera_names:
                        ret, frame = caps[cam].read()
                        if ret:
                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            img.thumbnail((896, 896), Image.Resampling.LANCZOS)
                            current_frames[cam] = img
                        else:
                            current_frames[cam] = Image.new('RGB', (896, 896), (0,0,0))
                    
                    frame_buffer.append(current_frames)

                    # Check if we have a full window and if it aligns with stride
                    window_start_idx = frame_idx - window_size + 1
                    
                    if len(frame_buffer) == window_size and window_start_idx >= 0 and window_start_idx % stride == 0:
                        # Construct Sample
                        window_imgs = []
                        w, h = 896, 896
                        cols, rows = grid
                        
                        for t in range(window_size):
                            canvas = Image.new('RGB', (w * cols, h * rows), (0,0,0))
                            frames_at_t = frame_buffer[t]
                            
                            for idx, cam in enumerate(camera_names):
                                if idx >= cols * rows: break
                                c_x, c_y = idx % cols, idx // cols
                                canvas.paste(frames_at_t[cam], (c_x * w, c_y * h))
                            window_imgs.append(canvas)

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
                        
                        # MICRO-FLUSH CHECK
                        if len(current_samples) >= FLUSH_THRESHOLD:
                            flush_to_disk()

                for c in caps.values(): c.release()
                for f in os.listdir(tmp_base): 
                    if f.endswith(".mp4"): os.remove(os.path.join(tmp_base, f))

            except Exception as e:
                log(f"Error processing clip {uuid}: {e}")
                traceback.print_exc()

        # Final flush for remaining samples
        flush_to_disk()

        for z in cam_zips.values(): z.close()
        ego_zip.close()
        
        # Merge all micro-shards
        if micro_shards:
            log(f"Merging {len(micro_shards)} micro-shards...")
            final_ds = concatenate_datasets([load_from_disk(p) for p in micro_shards])
            final_ds.save_to_disk(shard_path)
            log(f"Saved {len(final_ds)} samples to {shard_path}")
            
            # Cleanup temp dir
            try: shutil.rmtree(tmp_base) 
            except: pass
            
            return f"Batch {os.path.basename(shard_path)}: {len(final_ds)} samples."
        
        try: shutil.rmtree(tmp_base) 
        except: pass
        
        log("No samples generated.")
        return f"Batch {os.path.basename(shard_path)}: Empty."

    except Exception as e:
        # Catch and return the traceback so the main process knows WHY it crashed
        return f"CRASH in Worker {worker_id}: {str(e)}\n{traceback.format_exc()}"

# --- MAIN ORCHESTRATOR ---
def prepare():
    PROJECT_ROOT = "/workspace/AMD-Vision-Omni"
    data_cfg = f"{PROJECT_ROOT}/configs/data_config.yaml"
    ft_cfg = f"{PROJECT_ROOT}/configs/finetuning_config.yaml"
    SHARD_DIR = f"{PROJECT_ROOT}/data/shards"
    FINAL_OUTPUT = f"{PROJECT_ROOT}/data/processed_dataset"
    
    # Reduced workers to ensure stability during debugging
    NUM_WORKERS = 2
    CLIPS_PER_BATCH = 5

    # ARCHITECT FIX: Force 'spawn' to prevent OpenCV deadlocks
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(SHARD_DIR, exist_ok=True)
    print(f"--- [TURBO ETL: SAFE SPAWN MODE] ---")
    
    print("Scanning chunks...")
    raw_ds = NuRecTemporalDataset(data_cfg, ft_cfg, split='train')
    
    chunk_map = {}
    for s in raw_ds.samples:
        c_id = s['chunk']
        u_id = s['clip_uuid']
        if c_id not in chunk_map: chunk_map[c_id] = set()
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
    
    # Use ProcessPoolExecutor with the new spawn context
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

    print("\n--- [CONSOLIDATING] ---")
    all_batches = []
    for i in range(batch_idx):
        p = os.path.join(SHARD_DIR, f"batch_{i}")
        if os.path.exists(os.path.join(p, "dataset_info.json")):
            try: all_batches.append(load_from_disk(p))
            except: pass
            
    if all_batches:
        print(f"Merging {len(all_batches)} batches...")
        final_ds = concatenate_datasets(all_batches)
        final_ds.save_to_disk(FINAL_OUTPUT)
        print(f"✅ ETL COMPLETE: {len(final_ds)} samples saved.")
        shutil.rmtree(SHARD_DIR)
    else:
        print("❌ No data processed.")

if __name__ == "__main__":
    prepare()