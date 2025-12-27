import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config("configs/data_config.yaml")
local_dir = config['dataset']['local_dir']
filename = config['dataset']['index_file']
file_path = os.path.join(local_dir, filename)

df = pd.read_parquet(file_path)

total_clips = len(df)
total_chunks = df['chunk'].nunique()
splits = df['split'].value_counts()

print(f"Total Clips: {total_clips}")
print(f"Total Chunks: {total_chunks}")
print("Splits:")
print(splits)

# Analyze what we downloaded (Chunk 0)
chunk_0_clips = df[df['chunk'] == 0]
print(f"\nChunk 0 Clips: {len(chunk_0_clips)}")
print(f"Chunk 0 Split: {chunk_0_clips['split'].unique()}")
