import pandas as pd
import os
from huggingface_hub import hf_hub_download
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config("configs/data_config.yaml")
repo_id = config['dataset']['repo_id']
local_dir = config['dataset']['local_dir']
filename = config['dataset']['index_file']

# The file should already be downloaded by the previous run
file_path = os.path.join(local_dir, filename)

if not os.path.exists(file_path):
    print(f"File not found at {file_path}. Downloading...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        repo_type="dataset"
    )

df = pd.read_parquet(file_path)
print("Columns:", df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0])
print("\nFirst 5 rows head:")
print(df.head())
