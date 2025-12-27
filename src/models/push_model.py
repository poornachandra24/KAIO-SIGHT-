import torch
import os
from unsloth import FastVisionModel

# --- CONFIG ---
LOCAL_PATH = "checkpoints/AETHER-THINK-OMNI/checkpoint-60"
HF_REPO = "Thunderbird2410/AETHER-THINK-OMNI-Qwen2.5-VL-LoRA-Adapters" # New name to avoid confusion
# --------------

def push():
    print(f"📂 Loading weights from: {LOCAL_PATH}")
    
    # 1. Load the locally trained model
    model, tokenizer = FastVisionModel.from_pretrained(
        LOCAL_PATH,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )

    print(f"🚀 Uploading ONLY Adapters to: {HF_REPO}")
    
    # 2. Push Adapters (Native PEFT method)
    # This guarantees NO merging happens. It only uploads the LoRA layers.
    model.push_to_hub(HF_REPO)
    
    # 3. Push Tokenizer (Important for consistency)
    tokenizer.push_to_hub(HF_REPO)

    print("✅ Done! Size should be ~200MB.")

if __name__ == "__main__":
    push()