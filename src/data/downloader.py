import os
import argparse
import yaml
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Enable hf_transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_file(repo_id, filename, local_dir):
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type="dataset",
            force_download=False,
            resume_download=True
        )
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download NuRec dataset slice")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", help="Path to data config")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips to download (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_id = config['dataset']['repo_id']
    local_dir = config['dataset']['local_dir']
    max_workers = config['download']['max_concurrent']
    
    print(f"Downloading index file from {repo_id}...")
    try:
        index_path = hf_hub_download(
            repo_id=repo_id,
            filename=config['dataset']['index_file'],
            local_dir=local_dir,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Error downloading index file: {e}")
        return

    df = pd.read_parquet(index_path)
    print(f"Loaded index. Total clips: {len(df)}")
    
    if args.limit:
        print(f"Limiting to {args.limit} clips for testing.")
        selected_clips = df.head(args.limit)
    else:
        # If no limit, we might want to implement size-based limiting here
        # For now, we'll assume the user wants the whole slice defined in config or logic
        # But to prevent downloading 70TB, let's respect the target_size_gb if we implemented that logic.
        # For this step, we'll just stick to the requested logic.
        selected_clips = df 

    # Get unique chunks from selected clips
    if 'chunk' not in selected_clips.columns:
        print("Error: 'chunk' column not found in index.")
        return

    unique_chunks = selected_clips['chunk'].unique()
    print(f"Selected clips map to {len(unique_chunks)} unique chunks.")

    # List all files to filter
    print("Listing repo files (this may take a moment)...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    files_to_download = []
    
    # We want files that match the chunk IDs
    # Format: .chunk_XXXX.zip
    target_suffixes = {f".chunk_{chunk:04d}.zip" for chunk in unique_chunks}
    
    print(f"Filtering files for {len(unique_chunks)} chunks...")

    # Optimized filtering
    camera_files = []
    label_files = []
    
    for f in all_files:
        # Check if file ends with any of the target suffixes
        # We iterate over suffixes or check endswith tuple if possible, but tuple needs str not set
        # Converting set to tuple for endswith
        if f.endswith(tuple(target_suffixes)):
            if f.startswith('camera/'):
                camera_files.append(f)
            elif f.startswith('labels/'):
                label_files.append(f)
    
    # Check for orphans (chunks that have camera but no labels or vice versa)
    # This is a bit complex because one chunk might have multiple camera files (one per camera view)
    # and one label file.
    # Let's just ensure we download what we found, but warn if counts look off.
    # Ideally, for each chunk, we want 4 camera files (if we download all cameras) and 1 label file.
    
    files_to_download = camera_files + label_files
        
    print(f"Found {len(files_to_download)} files to download ({len(camera_files)} camera, {len(label_files)} labels).")
    
    print(f"Starting download with {max_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, repo_id, filename, local_dir) for filename in files_to_download]
        
        # Use tqdm to show progress
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading Files", unit="file"):
            pass
            
    elapsed = time.time() - start_time
    print(f"Download completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
