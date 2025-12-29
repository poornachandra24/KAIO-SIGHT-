import argparse
import os
import sys
import yaml

# Comet must be imported before torch/unsloth if possible, but here we add it at top
try:
    import comet_ml
except ImportError:
    comet_ml = None
import torch
from unsloth import FastVisionModel
from PIL import Image
import cv2
import numpy as np

# --- CONFIG DEFAULTS ---
DEFAULT_CONFIG = "configs/finetuning_config.yaml"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ADAPTER_ID = "Thunderbird2410/KAIO-SIGHT"

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_tiled_frames(uuid, data_dir, camera_list, grid, num_frames=16):
    """
    Reads 7 camera video files for the given UUID and tiles them into a single video sequence.
    """
    cols, rows = grid
    
    # 1. Open all video captures
    caps = {}
    frame_counts = []
    
    print(f"🎬 Opening {len(camera_list)} video streams for UUID: {uuid}...")
    
    for cam in camera_list:
        candidates = [
            os.path.join(data_dir, f"{uuid}.{cam}.mp4"),
            os.path.join(data_dir, f"{cam}.mp4"), 
        ]
        
        video_path = None
        for c in candidates:
            if os.path.exists(c):
                video_path = c
                break
        
        if not video_path:
            print(f"⚠️ Warning: File for {cam} not found. Using Placeholder.")
            caps[cam] = None
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open {video_path}. Using Placeholder.")
            caps[cam] = None
        else:
            caps[cam] = cap
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if not frame_counts:
        raise ValueError(f"No valid video files found for {uuid} in {data_dir}")
    
    total_frames = min(frame_counts) if frame_counts else 0
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    tiled_frames = []
    
    ref_w, ref_h = 1280, 720
    for cap in caps.values():
        if cap:
            ref_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ref_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            break
            
    # 2. Extract and Tile
    for i in range(total_frames):
        if i not in indices:
            for cap in caps.values():
                if cap: cap.read()
            continue
            
        current_imgs = []
        for cam in camera_list:
            cap = caps[cam]
            img = None
            if cap:
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if img is None:
                img = Image.new('RGB', (ref_w, ref_h), (0, 0, 0))
            
            current_imgs.append(img)
            
        canvas = Image.new('RGB', (ref_w * cols, ref_h * rows), (0, 0, 0))
        for idx, img in enumerate(current_imgs):
            if idx >= cols * rows: break
            c_x = idx % cols
            c_y = idx // cols
            canvas.paste(img, (c_x * ref_w, c_y * ref_h))
            
        tiled_frames.append(canvas)

    for cap in caps.values():
        if cap: cap.release()
        
    return tiled_frames

def save_video(frames, output_path, fps=4):
    """
    Saves a list of PIL images as an MP4 video.
    """
    if not frames: return
    w, h = frames[0].size
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        # PIL (RGB) -> OpenCV (BGR)
        img_np = np.array(frame)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    out.release()
    print(f"💾 Saved input video visualization to {output_path}")

def run():
    parser = argparse.ArgumentParser(description="Run inference with LoRA adapter on Multi-View Data")
    parser.add_argument("--uuid", type=str, required=True, help="Sample UUID (e.g., 25cd4769...)")
    parser.add_argument("--data_dir", type=str, default="data/samples", help="Directory containing the video files")
    parser.add_argument("--adapter_id", type=str, default=DEFAULT_ADAPTER_ID, help="HuggingFace adapter repo ID")
    parser.add_argument("--revision", type=str, default=None, help="Specific commit hash/revision")
    parser.add_argument("--no_lora", action="store_true", help="Run base model only")
    parser.add_argument("--track", action="store_true", help="Enable Comet ML tracking")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(DEFAULT_CONFIG)
    setup = cfg['vision']['camera_setup']
    camera_list = cfg['vision']['setups'][setup]['cameras']
    grid = cfg['vision']['setups'][setup]['grid']
    project_name = cfg.get('project_name', 'amd-vision-omni')

    # Initialize Comet
    experiment = None
    if args.track:
        if comet_ml:
            print(f"📊 Initializing Comet ML Experiment (Project: {project_name})")
            experiment = comet_ml.Experiment(
                project_name=project_name,
                auto_metric_logging=True,
                auto_output_logging="simple",
                log_code=True,
            )
            experiment.log_parameters({
                "uuid": args.uuid,
                "camera_setup": setup,
                "model": BASE_MODEL_ID,
                "adapter": args.adapter_id,
                "revision": args.revision or "latest"
            })
            experiment.add_tag("inference")
        else:
            print("⚠️ Comet ML requested but not installed. Skipping.")

    print(f"⚙️  Config: {setup} ({len(camera_list)} cameras, Grid: {grid})")

    # Load Frames
    frames = get_tiled_frames(args.uuid, args.data_dir, camera_list, grid)
    print(f"🎞️  Prepared {len(frames)} tiled frames.")

    # Logging Input Video
    if experiment:
        temp_vid = "inference_input.mp4"
        save_video(frames, temp_vid)
        experiment.log_video(temp_vid, name=f"input_{args.uuid}", overwrite=True)
        # Also log as an asset just in case
        # experiment.log_asset(temp_vid)

    # Load Model
    print(f"🚀 Loading Base Model: {BASE_MODEL_ID}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL_ID,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )
    
    if not args.no_lora:
        print(f"🔗 Attaching LoRA: {args.adapter_id} (Revision: {args.revision if args.revision else 'Latest'})...")
        model.load_adapter(args.adapter_id, revision=args.revision)
    
    FastVisionModel.for_inference(model) 

    # Prepare Input
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": f"Analyze the {setup} sequence. Predict ego-motion."}
        ]}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], videos=[frames], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    # Generate
    print("🧠 Thinking...")
    try:
        outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True, temperature=0.2)
        
        # Decode
        decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"\n✅ PREDICTION:\n{decoded}")

        if experiment:
            experiment.log_text(decoded, metadata={"type": "prediction"})
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        if experiment: experiment.log_text(str(e), metadata={"type": "error"})
        raise e
    finally:
        if experiment:
            # Check for temp file
            if os.path.exists("inference_input.mp4"):
                os.remove("inference_input.mp4")
            experiment.end()

if __name__ == "__main__":
    run()