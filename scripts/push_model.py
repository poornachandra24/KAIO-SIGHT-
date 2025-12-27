import torch
import os
from unsloth import FastVisionModel

# --- ⚙️ CONFIGURATION ⚙️ ---
# The path where your training finished
LOCAL_MODEL_PATH = "checkpoints/AETHER-THINK-OMNI/checkpoint-60"

# Your Hugging Face Repo (Pre-filled for you)
HF_REPO_ID = "Thunderbird2410/AETHER-THINK-OMNI-Qwen2.5-VL-LoRA"

# Set True to upload the full 15GB model (Merged).
# Set False to upload only the small LoRA adapters (Fast).
PUSH_FULL_MODEL = False 
# ---------------------------

def push():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ Error: Path not found: {LOCAL_MODEL_PATH}")
        return

    print(f"📂 Loading model from: {LOCAL_MODEL_PATH}")
    
    # Load weights
    model, tokenizer = FastVisionModel.from_pretrained(
        LOCAL_MODEL_PATH,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )

    print(f"🚀 Pushing to Hub: {HF_REPO_ID}")

    if PUSH_FULL_MODEL:
        print("⏳ Merging weights (this takes RAM)...")
        model.push_to_hub_merged(
            HF_REPO_ID,
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        print("⚡ Uploading LoRA Adapters only (Recommended)...")
        model.push_to_hub_merged(
            HF_REPO_ID,
            tokenizer,
            save_method="lora",
        )

    print(f"✅ Success! View here: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    push()