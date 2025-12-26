import os
import yaml
import zipfile
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_configs():
    with open("configs/data_config.yaml", 'r') as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/finetuning_config.yaml", 'r') as f:
        ft_cfg = yaml.safe_load(f)
    return data_cfg, ft_cfg

def audit_physical_data():
    data_cfg, ft_cfg = load_configs()
    local_dir = data_cfg['dataset']['local_dir']
    
    # Configuration to check against
    setup_key = ft_cfg['vision']['camera_setup']
    req_cams = ft_cfg['vision']['setups'][setup_key]['cameras']
    window_size = ft_cfg['vision']['window_size']
    
    print(f"--- [ARCHITECTURAL DATA AUDIT: {setup_key}] ---")
    
    report = []
    report.append(f"# Physical AI Data Audit: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"- **Data Root**: `{local_dir}`")
    report.append(f"- **Target Configuration**: `{setup_key}` ({len(req_cams)} cameras)")
    
    # 1. Discover Chunks
    # We use the first camera folder to find downloaded chunks
    anchor_cam = req_cams[0]
    anchor_path = Path(local_dir) / "camera" / anchor_cam
    
    if not anchor_path.exists():
        return print(f"❌ Error: Anchor camera folder {anchor_path} missing.")

    found_zips = list(anchor_path.glob("*.zip"))
    chunk_ids = []
    for f in found_zips:
        try:
            chunk_ids.append(int(f.name.split(".chunk_")[1].split(".zip")[0]))
        except: continue
    
    # 2. Integrity Check (Synchronization)
    synced_chunks = []
    missing_data = []
    
    for cid in sorted(chunk_ids):
        c_str = f"{cid:04d}"
        missing_cams = []
        for cam in req_cams:
            p = Path(local_dir) / "camera" / cam / f"{cam}.chunk_{c_str}.zip"
            if not p.exists():
                missing_cams.append(cam)
        
        ego_p = Path(local_dir) / "labels" / "egomotion" / f"egomotion.chunk_{c_str}.zip"
        ego_exists = ego_p.exists()
        
        if not missing_cams and ego_exists:
            synced_chunks.append(cid)
        else:
            missing_data.append({"chunk": c_str, "missing_cams": missing_cams, "ego_missing": not ego_exists})

    # 3. Sample Estimation (High-Precision Audit)
    total_samples = 0
    total_size_bytes = 0
    
    for cid in synced_chunks:
        c_str = f"{cid:04d}"
        ego_p = Path(local_dir) / "labels" / "egomotion" / f"egomotion.chunk_{c_str}.zip"
        
        # Check internal format and count frames
        with zipfile.ZipFile(ego_p, 'r') as z:
            parquet_files = [f for f in z.namelist() if f.endswith('.parquet')]
            # Assume first parquet represents clip length
            if parquet_files:
                with z.open(parquet_files[0]) as f:
                    import io
                    df = pd.read_parquet(io.BytesIO(f.read()))
                    frames_in_chunk = len(df)
                    # sliding window calculation
                    samples_in_chunk = max(0, (frames_in_chunk - window_size) // (window_size // 2))
                    total_samples += samples_in_chunk
        
        # Calculate size for this chunk
        for cam in req_cams:
            total_size_bytes += (Path(local_dir) / "camera" / cam / f"{cam}.chunk_{c_str}.zip").stat().st_size
        total_size_bytes += ego_p.stat().st_size

    # 4. Finalizing Report
    report.append(f"\n## Integrity Summary")
    report.append(f"- **Chunks Found**: {len(chunk_ids)}")
    report.append(f"- **Fully Synchronized Chunks**: {len(synced_chunks)}")
    report.append(f"- **Orphaned/Incomplete Chunks**: {len(missing_data)}")
    
    report.append(f"\n## Training Capacity")
    report.append(f"- **Estimated Total Training Samples**: {total_samples}")
    report.append(f"- **Total Synchronized Data Size**: {total_size_bytes / (1024**3):.2f} GB")
    
    if missing_data:
        report.append(f"\n## Missing Data Alert")
        report.append("| Chunk | Status |")
        report.append("| :--- | :--- |")
        for m in missing_data[:10]:
            msg = f"Missing: {', '.join(m['missing_cams'])}" if m['missing_cams'] else "Missing Labels"
            report.append(f"| {m['chunk']} | {msg} |")

    # Output to File
    output_dir = "docs/data_audit"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/audit_report.md", "w") as f:
        f.write("\n".join(report))
        
    print(f"✅ Audit Complete. Samples found: {total_samples}. Sync Check: {len(synced_chunks)}/{len(chunk_ids)}")
    print(f"Report saved to {output_dir}/audit_report.md")

if __name__ == "__main__":
    audit_physical_data()