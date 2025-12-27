from unsloth import FastVisionModel
import torch
from PIL import Image
import cv2
import numpy as np

# --- CONFIG ---
MODEL_ID = "Thunderbird2410/AETHER-THINK-OMNI-Qwen2.5-VL-LoRA"
# UPDATE THIS LINE
VIDEO_PATH = "data/samples/25cd4769-5dcf-4b53-a351-bf2c5deb6124.camera_cross_right_120fov.mp4"
# --------------

def get_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle potentially broken metadata
    if total_frames <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        final_frames = [frames[i] for i in indices]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        final_frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                final_frames.append(Image.fromarray(frame))
                
    cap.release()
    return final_frames

def run():
    # 1. Check Hardware
    # On ROCm, torch.cuda.is_available() returns True if the MI300X is visible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Detected Device: {torch.cuda.get_device_name(0)} (mapped to '{device}')")

    print(f"📂 Loading Model: {MODEL_ID}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )
    FastVisionModel.for_inference(model) 

    # 2. Process Video
    print(f"🎬 Processing video: {VIDEO_PATH}")
    frames = get_video_frames(VIDEO_PATH)
    print(f"✅ Extracted {len(frames)} frames.")
    
    instruction = "Analyze the 4-cam sequence. Predict ego-motion."
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": instruction}
        ]}
    ]

    # 3. FIX: Two-Step Processing
    # Step A: Create the prompt string (Text)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Step B: Turn Text + Video Images into Tensors
    # This processor handles the visual embeddings
    inputs = tokenizer(
        text=[text],
        videos=[frames],
        padding=True,
        return_tensors="pt",
    )
    
    # Move to MI300X
    inputs = inputs.to(device)

    # 4. Generate
    print("🧠 Thinking...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.2,
    )
    
    # 5. Decode
    print("\n✅ PREDICTION:")
    # Filter out the input tokens to show only the new prediction
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

if __name__ == "__main__":
    run()