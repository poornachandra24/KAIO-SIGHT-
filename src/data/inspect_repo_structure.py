from huggingface_hub import list_repo_files
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config("configs/data_config.yaml")
repo_id = config['dataset']['repo_id']

print(f"Listing files in {repo_id}...")
# We can't easily limit list_repo_files without fetching all, but we can iterate if it returns a generator (it returns a list usually).
# Let's just get all and print the first few that start with 'camera/'
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

print(f"Total files: {len(all_files)}")

camera_files = [f for f in all_files if f.startswith('camera/')]
print(f"Camera files found: {len(camera_files)}")

print("\nFirst 20 camera files:")
for f in camera_files[:20]:
    print(f)

print("\nFirst 20 label files:")
label_files = [f for f in all_files if f.startswith('labels/')]
for f in label_files[:20]:
    print(f)
