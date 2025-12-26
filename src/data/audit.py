import os
import yaml
import zipfile
import pandas as pd
from pathlib import Path
from datetime import datetime
import io

def load_configs():
    with open("configs/data_config.yaml", 'r') as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/finetuning_config.yaml", 'r') as f:
        ft_cfg = yaml.safe_load(f)
    return data_cfg, ft_cfg

def audit_physical_data():
    data_cfg, ft_cfg = load_configs()
    local_dir = data_cfg['dataset']['local_dir']
    
    setup_key = ft_cfg['vision']['camera_setup']
    req_cams = ft_cfg['vision']['setups'][setup_key]['cameras']
    window_size = ft_cfg['vision']['window_size']
    
    print(f"--- [ARCHITECTURAL DATA AUDIT: {setup_key}] ---")
    
    # 1. Discover Chunks
    anchor_cam = req_cams[0]
    anchor_path = Path(local_dir) / "camera" / anchor_cam
    
    if not anchor_path.exists():
        print(f"❌ Error: Anchor camera folder {anchor_path} missing.")
        return 0, 0, 0

    found_zips = list(anchor_path.glob("*.zip"))
    chunk_ids = []
    for f in found_zips:
        try:
            chunk_ids.append(int(f.name.split(".chunk_")[1].split(".zip")[0]))
        except: continue
    
    # 2. Integrity Check
    synced_chunks = []
    for cid in sorted(chunk_ids):
        c_str = f"{cid:04d}"
        missing_cams = False
        for cam in req_cams:
            if not (Path(local_dir) / "camera" / cam / f"{cam}.chunk_{c_str}.zip").exists():
                missing_cams = True
                break
        
        ego_p = Path(local_dir) / "labels" / "egomotion" / f"egomotion.chunk_{c_str}.zip"
        if not missing_cams and ego_p.exists():
            synced_chunks.append(cid)

    # 3. Sample Estimation
    total_samples = 0
    total_size_bytes = 0
    
    for cid in synced_chunks:
        c_str = f"{cid:04d}"
        ego_p = Path(local_dir) / "labels" / "egomotion" / f"egomotion.chunk_{c_str}.zip"
        
        try:
            with zipfile.ZipFile(ego_p, 'r') as z:
                parquet_files = [f for f in z.namelist() if f.endswith('.parquet')]
                if parquet_files:
                    with z.open(parquet_files[0]) as f:
                        df = pd.read_parquet(io.BytesIO(f.read()))
                        frames_in_chunk = len(df)
                        total_samples += max(0, (frames_in_chunk - window_size) // (window_size // 2))
            
            for cam in req_cams:
                total_size_bytes += (Path(local_dir) / "camera" / cam / f"{cam}.chunk_{c_str}.zip").stat().st_size
            total_size_bytes += ego_p.stat().st_size
        except: continue

    # Output human-readable summary
    print(f"✅ Audit Complete. Synced Chunks: {len(synced_chunks)}")
    
    # Return metrics for the machine-readable block
    return len(synced_chunks), total_samples, total_size_bytes

if __name__ == "__main__":
    # Execute audit and capture results
    synced, samples, size_bytes = audit_physical_data()
    
    # --- MACHINE READABLE BLOCK FOR SHELL SCRIPT ---
    print(f"---METRICS---")
    print(f"SYNCED_CHUNKS={synced}")
    print(f"TOTAL_SAMPLES={samples}")
    print(f"DATA_SIZE_GB={size_bytes / (1024**3):.2f}")