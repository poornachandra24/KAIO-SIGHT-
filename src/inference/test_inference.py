from unsloth import FastVisionModel
import torch
from PIL import Image
import cv2
import numpy as np

# --- CONFIG ---
# 1. The Official Base Model
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# 2. YOUR Adapter Repo (The small ~200MB one)
ADAPTER_ID = "Thunderbird2410/AETHER-THINK-OMNI-Qwen2.5-VL-LoRA-Adapters"

VIDEO_PATH = "data/samples/25cd4769-5dcf-4b53-a351-bf2c5deb6124.camera_cross_right_120fov.mp4" 
# --------------

def get_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Bad video path: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Simple extraction logic
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if i in indices and ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def run():
    print(f"🚀 Loading Base Model: {BASE_MODEL_ID}...")
    # 1. Load Base Model (Pure Qwen)
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL_ID,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )
    
    # 2. Load Your Training (The Delta)
    print(f"🔗 Attaching LoRA: {ADAPTER_ID}...")
    model.load_adapter(ADAPTER_ID)

    # 3. Optimize
    FastVisionModel.for_inference(model) 

    # 4. Process Inputs
    print(f"🎬 Processing video...")
    frames = get_video_frames(VIDEO_PATH)
    
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": "Analyze the 4-cam sequence. Predict ego-motion."}
        ]}
    ]

    # Prepare inputs using the tokenizer/processor
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], videos=[frames], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    # 5. Generate
    print("🧠 Thinking...")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.2)
    print("\n Raw Output :\n", outputs)
    # 6. Decode
    decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"\n✅ PREDICTION:\n{decoded}")

if __name__ == "__main__":
    run()