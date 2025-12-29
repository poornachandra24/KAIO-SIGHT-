import argparse
import os
import zipfile
import pandas as pd
import io
import numpy as np
import shutil
import yaml
import random
import sys

# Defaults
DEFAULT_DATA_DIR = "/workspace/KAIO-SIGHT/data"
DEFAULT_CONFIG = "configs/finetuning_config.yaml"

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_available_uuids(data_dir, camera_name="camera_front_wide_120fov"):
    """
    Scans the zip files for the specified camera to find available sample UUIDs.
    """
    cam_dir = os.path.join(data_dir, "camera", camera_name)
    if not os.path.exists(cam_dir):
        print(f"⚠️  Camera dir not found: {cam_dir}")
        return []

    uuids = set()
    zip_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.zip')])
    
    print(f"🔍 Scanning {len(zip_files)} chunks in {camera_name}...")
    
    # Scan first 3 chunks to get a good candidate pool without being too slow
    for zip_name in zip_files[:3]:
        zip_path = os.path.join(cam_dir, zip_name)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Format: {uuid}.{camera_name}.mp4
                # We filter for mp4s
                suffix = f".{camera_name}.mp4"
                for f in z.namelist():
                    if f.endswith(suffix):
                        uuid = f[:-len(suffix)]
                        uuids.add(uuid)
        except Exception as e:
            print(f"  ❌ Error reading {zip_name}: {e}")
            
    return list(uuids)

def extract_videos(uuid, data_dir, output_dir, camera_list):
    """
    Extracts video files for a given UUID from the camera zip archives.
    """
    print(f"📂 Extracting videos for {uuid}...")
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    for cam in camera_list:
        found = False
        cam_dir = os.path.join(data_dir, "camera", cam)
        if not os.path.exists(cam_dir):
            print(f"⚠️  Camera dir not found: {cam_dir}")
            continue

        zip_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.zip')])
        
        for zip_name in zip_files:
            zip_path = os.path.join(cam_dir, zip_name)
            target_file = f"{uuid}.{cam}.mp4"
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    if target_file in z.namelist():
                        z.extract(target_file, output_dir)
                        extracted_count += 1
                        print(f"  ✅ Extracted {target_file}")
                        found = True
                        break
            except Exception as e:
                print(f"  ❌ Error reading {zip_path}: {e}")
        
        if not found:
            print(f"  ⚠️  Could not find video for {cam}")

    print(f"🏁 Extracted {extracted_count}/{len(camera_list)} videos.\n")

def get_ground_truth(uuid, data_dir, window_size=16):
    """
    Extracts egomotion ground truth and calculates displacement/velocity.
    """
    print(f"📊 Calculating Ground Truth for {uuid}...")
    
    labels_dir = os.path.join(data_dir, "labels", "egomotion")
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return

    zip_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.zip')])
    parquet_file = f"{uuid}.egomotion.parquet"
    
    df = None
    for zip_name in zip_files:
        zip_path = os.path.join(labels_dir, zip_name)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                if parquet_file in z.namelist():
                    with z.open(parquet_file) as f:
                        df = pd.read_parquet(io.BytesIO(f.read()))
                    break
        except Exception:
            continue
            
    if df is None:
        print(f"❌ Ground truth file {parquet_file} not found in any zip.")
        return

    # Sort by timestamp
    time_col = next((c for c in ['timestamp_us', 'timestamp_ns', 'timestamp'] if c in df.columns), df.columns[0])
    df = df.sort_values(time_col)
    
    # Process Window
    window = df.iloc[:window_size]
    
    if len(window) < 2:
        print("⚠️ Not enough frames for calculation.")
        return

    # 1. Displacement
    p_start = np.array(window.iloc[0][['x', 'y']].values, dtype=float)
    p_end = np.array(window.iloc[-1][['x', 'y']].values, dtype=float)
    dist = np.linalg.norm(p_end - p_start)
    
    # 2. Average Velocity
    vels = [np.linalg.norm([r['vx'], r['vy']]) for _, r in window.iterrows()]
    v_avg = np.mean(vels)
    
    print("-" * 30)
    print(f"✅ GROUND TRUTH (First {len(window)} frames)")
    print(f"   UUID         : {uuid}")
    print(f"   Displacement : {dist:.2f} m")
    print(f"   Avg Velocity : {v_avg:.2f} m/s")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Prepare specific sample for inference testing")
    parser.add_argument("--uuid", type=str, help="Specific Sample UUID to extract")
    parser.add_argument("--list", action="store_true", help="List available UUIDs")
    parser.add_argument("--random", action="store_true", help="Pick a random UUID")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="data/samples", help="Where to save extracted videos")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to finetuning config")
    
    args = parser.parse_args()

    # Load Config for Camera Setup
    cfg = load_config(args.config)
    setup = cfg['vision']['camera_setup']
    camera_list = cfg['vision']['setups'][setup]['cameras']
    
    # Handle Scenarios
    if args.list:
        uuids = get_available_uuids(args.data_dir, camera_list[0])
        print(f"📋 Available UUIDs ({len(uuids)} found):")
        for u in uuids[:20]:
            print(f"  - {u}")
        print("... (use --uuid <ID> to select one)")
        return

    if args.random:
        uuids = get_available_uuids(args.data_dir, camera_list[0])
        if not uuids:
            print("❌ No UUIDs found.")
            return
        args.uuid = random.choice(uuids)
        print(f"🎲 Randomly selected UUID: {args.uuid}")
    
    if not args.uuid:
        print("❌ Error: Must specify --uuid, --list, or --random")
        parser.print_help()
        return

    # 1. Extract Videos
    extract_videos(args.uuid, args.data_dir, args.output_dir, camera_list)
    
    # 2. Get Ground Truth
    get_ground_truth(args.uuid, args.data_dir)

if __name__ == "__main__":
    main()
