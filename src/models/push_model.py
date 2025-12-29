import torch
import os
import glob
import yaml
from unsloth import FastVisionModel
from huggingface_hub import HfApi

# --- CONFIG ---
# --- CONFIG ---
CONFIG_PATH = "configs/finetuning_config.yaml"
# --------------

def get_latest_report(report_dir="docs/reports"):
    """Finds the most recently created run directory and its report."""
    runs = sorted(glob.glob(os.path.join(report_dir, "run_*")))
    if not runs:
        return None
    latest_run = runs[-1]
    report_path = os.path.join(latest_run, "report.md")
    if os.path.exists(report_path):
        return report_path
    return None

def push():
    # 1. Load Config
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    project_name = cfg.get('project_name', "AETHER-THINK-OMNI-v1-metrics")
    hf_repo = cfg.get('hub', {}).get('hf_repo', "Thunderbird2410/KAIO-SIGHT")
    
    # 2. Resolve Checkpoint Path
    base_ckpt_dir = os.path.join("checkpoints", project_name)
    if not os.path.exists(base_ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {base_ckpt_dir}")
    
    # Find all checkpoint directories
    checkpoints = sorted(glob.glob(os.path.join(base_ckpt_dir, "checkpoint-*")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_ckpt_dir}")
        
    # Pick the latest one (assumes lexical sort works for checkpoint-X, or use modification time)
    # Using modification time is safer for non-padded numbers if we want purely latest created
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    print(f"📂 Resolved latest checkpoint: {latest_checkpoint}")
    
    # 3. Load the locally trained model
    model, tokenizer = FastVisionModel.from_pretrained(
        latest_checkpoint,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )

    print(f"🚀 Uploading ONLY Adapters to: {hf_repo}")
    
    # 3. Push Adapters (Native PEFT method)
    # This guarantees NO merging happens. It only uploads the LoRA layers.
    commit_msg = f"Upload adapters for project: {project_name}"
    commit_info = model.push_to_hub(hf_repo, commit_message=commit_msg)
    
    commit_hash = "unknown"
    if hasattr(commit_info, 'oid'):
        commit_hash = commit_info.oid
    
    # Fallback: If push_to_hub returns None (e.g. no changes or library quirk), fetch from API
    if commit_hash == "unknown":
        print("⚠️ push_to_hub returned None or no oid. Fetching latest commit from Hub API...")
        try:
            api = HfApi()
            repo_info = api.model_info(hf_repo)
            commit_hash = repo_info.sha
            print(f"✅ Fetched latest commit hash: {commit_hash}")
        except Exception as e:
            print(f"❌ Failed to fetch commit hash: {e}")

    # 4. Push Tokenizer (Important for consistency)
    tokenizer.push_to_hub(hf_repo)

    print(f"✅ Done! Size should be ~200MB. Commit: {commit_hash}")

    # 5. Update Report
    report_path = get_latest_report()
    if report_path:
        print(f"📝 Updating report at: {report_path}")
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n## 🚀 HuggingFace Deployment\n")
            f.write(f"**Repo:** [{hf_repo}](https://huggingface.co/{hf_repo})\n")
            f.write(f"**Commit:** `{commit_hash}`\n")
            f.write(f"**Adapter Config:** [adapter_config.json](https://huggingface.co/{hf_repo}/blob/{commit_hash}/adapter_config.json)\n")
    else:
        print("⚠️ No report found to update.")

if __name__ == "__main__":
    push()