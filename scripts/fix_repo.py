from huggingface_hub import hf_hub_download, upload_file

# Your Repo
MY_REPO = "Thunderbird2410/AETHER-THINK-OMNI-Qwen2.5-VL-LoRA"
# The Base Model we copied config from
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"🔧 Repairing {MY_REPO}...")

# 1. Config.json (Architecture definition)
print("⬇️  Fetching config.json...")
config_path = hf_hub_download(repo_id=BASE_MODEL, filename="config.json")
upload_file(path_or_fileobj=config_path, path_in_repo="config.json", repo_id=MY_REPO)

# 2. Generation Config (Inference settings)
print("⬇️  Fetching generation_config.json...")
gen_path = hf_hub_download(repo_id=BASE_MODEL, filename="generation_config.json")
upload_file(path_or_fileobj=gen_path, path_in_repo="generation_config.json", repo_id=MY_REPO)

print("✅ Repository fixed! Now 'test_inference.py' will work.")