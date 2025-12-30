import os
import subprocess
import re
import pandas as pd
import numpy as np
import argparse
import sys
import zipfile
import io

# Add current directory to path to allow imports if needed, 
# but we will reimplement ground truth logic for stability.

UUID = "402fdeb6-f078-44af-a9b3-79bfa15961a1"
DATA_DIR = "data/samples"
REPORTS_DIR = "docs/reports"

# Verified commits from reports
COMMITS = [
    "c7e9029e40204d9ea3bceb1e2465be50a80da10e",
    "098d1b84979da84da48c3586f4d5f442ff827111",
    "71cd2ddf5714cc2ad7310ac1124d49c70842d9ae",
    "07ed155ded98742dce5f3973cc60919b634198ae",
    "57c7030ea38a5a2f24e4116300055142f3373c08",
    "9b160f7b8e22607f341cdce3aa8376f5652688a2",
    "435ddfe3a14f605d1c0915b71b67156dea15a618",
    "91dbf5834c14434dc9f9b4c8c2fa242d35ab66eb",
    "bc0b3c66c865427da4fc73b49bf9154f0829f597",
    "2f02e5656a29074fa671bbc561b0ac54a4437e6b"
]

def get_ground_truth(uuid, base_data_dir="data", window_size=16):
    """
    Calculates Ground Truth directly from the label zip files.
    """
    labels_dir = os.path.join(base_data_dir, "labels", "egomotion")
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return None, None

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
        return None, None

    # Sort by timestamp
    time_col = next((c for c in ['timestamp_us', 'timestamp_ns', 'timestamp'] if c in df.columns), df.columns[0])
    df = df.sort_values(time_col)
    
    # Process Window
    window = df.iloc[:window_size]
    
    if len(window) < 2:
        return None, None

    # 1. Displacement
    p_start = np.array(window.iloc[0][['x', 'y']].values, dtype=float)
    p_end = np.array(window.iloc[-1][['x', 'y']].values, dtype=float)
    dist = np.linalg.norm(p_end - p_start)
    
    # 2. Average Velocity
    vels = [np.linalg.norm([r['vx'], r['vy']]) for _, r in window.iterrows()]
    v_avg = np.mean(vels)
    
    return dist, v_avg

def parse_prediction(text):
    """
    Parses the text output from the model to find numerical predictions.
    Regex adapted to potential model outputs.
    """
    # Pattern for "Displacement: 1.23 m" or "1.23m"
    disp_match = re.search(r"Displacement[:\s]+([\d\.]+)\s*m?", text, re.IGNORECASE)
    # Pattern for "Velocity: 4.56 m/s" or "Speed: ..."
    vel_match = re.search(r"(?:Velocity|Speed)[:\s]+([\d\.]+)\s*m/s?", text, re.IGNORECASE)
    
    d = float(disp_match.group(1)) if disp_match else None
    v = float(vel_match.group(1)) if vel_match else None
    return d, v

def run_evaluation():
    print(f"🚀 Starting Evaluation for UUID: {UUID}")
    
    # 1. Get Ground Truth
    gt_disp, gt_vel = get_ground_truth(UUID, base_data_dir="data")
    if gt_disp is None:
        print("❌ Could not calculate Ground Truth. Aborting.")
        return
        
    print(f"✅ Ground Truth -> Displacement: {gt_disp:.4f} m, Velocity: {gt_vel:.4f} m/s")
    
    results = []
    
    # 2. Iterate Commits
    for idx, commit in enumerate(COMMITS):
        print(f"\n[{idx+1}/{len(COMMITS)}] Testing Commit: {commit}")
        
        cmd = [
            "python", "src/inference/test_inference.py",
            "--uuid", UUID,
            "--revision", commit,
            "--track", # Enable Comet tracking
            "--project_name", "kaio-sight-inference-test"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600) # 10 min timeout per run
            
            output = result.stdout
            
            # Extract Experiment Name
            exp_name_match = re.search(r"COMET_EXPERIMENT_NAME: (.+)", output)
            if not exp_name_match:
                 # Fallback to standard Comet logging format
                 exp_name_match = re.search(r"COMET INFO:\s+name\s+:\s+(.+)", output)
            
            exp_name = exp_name_match.group(1).strip() if exp_name_match else "N/A"

            # Extract Experiment URL
            exp_url_match = re.search(r"(https?://www\.comet\.com/[^\s]+)", output)
            exp_url = exp_url_match.group(1).strip() if exp_url_match else "N/A"
            
            # Extract Prediction Text
            # Capture everything from PREDICTION start until the Hardware Stats section or End of String
            pred_match = re.search(r"✅ PREDICTION:\n(.*?)(\n📊 INFERENCE HARDWARE STATS:|\Z)", output, re.DOTALL)
            pred_text = pred_match.group(1).strip() if pred_match else "No prediction found"
            
            # Parse metrics
            pred_disp, pred_vel = parse_prediction(pred_text)
            
            # Errors
            err_disp = abs(pred_disp - gt_disp) if pred_disp is not None else None
            err_vel = abs(pred_vel - gt_vel) if pred_vel is not None else None
            
            print(f"   prediction: {pred_text}")
            print(f"   disp_err: {err_disp}, vel_err: {err_vel}")
            print(f"   comet_name: {exp_name}")
            print(f"   comet_url: {exp_url}")
            
            results.append({
                "commit_id": commit,
                "ground_truth_disp": gt_disp,
                "ground_truth_vel": gt_vel,
                "pred_disp": pred_disp,
                "pred_vel": pred_vel,
                "error_disp": err_disp,
                "error_vel": err_vel,
                "raw_output": pred_text,
                "comet_experiment_name": exp_name,
                "comet_url": exp_url
            })
            
            # Incremental Save
            output_path = "src/inference/evaluation_results.parquet"
            df = pd.DataFrame(results)
            if 'error_disp' in df.columns:
                df = df.sort_values("error_disp")
            df.to_parquet(output_path)
            print(f"   💾 Updated {output_path} ({len(df)} records)")
            
        except subprocess.TimeoutExpired:
            print("   ❌ Timeout expired.")
            results.append({"commit_id": commit, "error": "Timeout"})
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({"commit_id": commit, "error": str(e)})

    # Final Summary
    output_path = "src/inference/evaluation_results.parquet"
    df = pd.DataFrame(results)
    if 'error_disp' in df.columns:
        df = df.sort_values("error_disp")
    
    print(f"\n✅ Evaluation Complete via {output_path}")
    print(df[['commit_id', 'pred_disp', 'error_disp', 'comet_experiment_name']])

if __name__ == "__main__":
    run_evaluation()
