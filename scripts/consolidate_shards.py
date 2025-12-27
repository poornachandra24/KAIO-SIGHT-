#!/usr/bin/env python3
"""
Simple consolidation - NO concatenation, just organize existing shards.
"""
import os
import shutil
import json
from datasets import load_from_disk

SHARD_DIR = "/workspace/AMD-Vision-Omni/data/shards"
FINAL_OUTPUT = "/workspace/AMD-Vision-Omni/data/processed_dataset"

print("Organizing shards (no concatenation)...")

if os.path.exists(FINAL_OUTPUT):
    print(f"Cleaning existing output: {FINAL_OUTPUT}")
    shutil.rmtree(FINAL_OUTPUT)
os.makedirs(FINAL_OUTPUT, exist_ok=True)

shard_idx = 0
total_samples = 0

# Walk through batch directories
for batch_dir in sorted(os.listdir(SHARD_DIR)):
    batch_path = os.path.join(SHARD_DIR, batch_dir)
    if not os.path.isdir(batch_path):
        continue
    
    # Walk through part directories (micro-shards)
    for part_dir in sorted(os.listdir(batch_path)):
        if part_dir.startswith("part_"):
            src = os.path.join(batch_path, part_dir)
            dst = os.path.join(FINAL_OUTPUT, f"shard_{shard_idx:04d}")
            
            # Move instead of copy to save disk space
            shutil.move(src, dst)
            
            # Count samples
            try:
                # Try to load dataset info to get count
                ds = load_from_disk(src)
                count = len(ds)
            except:
                print(f"  Warning: Could not load {src}, assuming 30 samples")
                count = 30  # default fallback
                
            total_samples += count
            
            if shard_idx % 100 == 0:
                print(f"  Processed {shard_idx} shards...")
            shard_idx += 1

# Create manifest
manifest = {
    "total_samples": total_samples,
    "num_shards": shard_idx,
    "note": "Load with: datasets.load_from_disk('data/processed_dataset/shard_XXXX')"
}
with open(os.path.join(FINAL_OUTPUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n✅ DONE! Organized {shard_idx} shards (~{total_samples} samples)")
print(f"   Location: {FINAL_OUTPUT}")
