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


import re
import time
import threading
import subprocess
import queue

class HardwareMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.metrics = {
            "vram_used": [], "power_avg": [], "temp_edge": [], "temp_junc": []
        }
        self.thread = threading.Thread(target=self._monitor_loop)

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            try:
                cmd = ["rocm-smi", "--showmeminfo", "vram", "--showpower", "--showtemp"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0: self._parse_smi(result.stdout)
            except Exception as e: print(f"Monitor Warning: {e}")
            time.sleep(self.interval)

    def _parse_smi(self, output):
        vram = re.search(r"VRAM Total Used Memory \(B\): (\d+)", output)
        if vram: self.metrics["vram_used"].append(int(vram.group(1)) / 1024 / 1024)
        pwr = re.search(r"Average Power \(W\): ([\d\.]+)", output)
        if pwr: self.metrics["power_avg"].append(float(pwr.group(1)))
        te = re.search(r"Temperature \(Sensor edge\) \(C\): ([\d\.]+)", output)
        if te: self.metrics["temp_edge"].append(float(te.group(1)))
        tj = re.search(r"Temperature \(Sensor junction\) \(C\): ([\d\.]+)", output)
        if tj: self.metrics["temp_junc"].append(float(tj.group(1)))

    def get_summary(self):
        summary = {}
        for k, v in self.metrics.items():
            if v:
                summary[f"{k}_min"] = min(v)
                summary[f"{k}_max"] = max(v)
                summary[f"{k}_avg"] = sum(v) / len(v)
        return summary

class AsyncVideoReader:
    """Reads frames from a video file in a separate thread to prevent IO blocking"""
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.is_valid = self.cap.isOpened()
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_valid else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.is_valid else 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.is_valid else 720
        self.queue = queue.Queue(maxsize=32)
        self.running = False
        self.thread = None

    def start(self, indices):
        if not self.is_valid: return
        self.running = True
        self.thread = threading.Thread(target=self._worker, args=(indices,))
        self.thread.start()

    def _worker(self, indices):
        current_idx = 0
        indices_set = set(indices)
        max_idx = max(indices) if len(indices) > 0 else 0
        
        while self.running and current_idx <= max_idx:
            if current_idx in indices_set:
                ret, frame = self.cap.read()
                if not ret: break
                # Convert BGR->RGB in worker thread to save main thread time
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.queue.put((current_idx, frame))
            else:
                # Skip frame
                self.cap.read()
            current_idx += 1
        self.queue.put(None) # Sentinel

    def get_frame(self):
        if not self.is_valid: return None
        item = self.queue.get()
        if item is None: return None
        return item[1]

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.cap: self.cap.release()

def get_tiled_frames(uuid, data_dir, camera_list, grid, num_frames=4):
    cols, rows = grid
    
    # Identify files
    readers = {}
    valid_readers = []
    
    print(f"🎬 Initializing threaded readers for {len(camera_list)} cameras...")
    
    for cam in camera_list:
        path = os.path.join(data_dir, f"{uuid}.{cam}.mp4")
        if not os.path.exists(path):
             path = os.path.join(data_dir, f"{cam}.mp4")
        
        reader = AsyncVideoReader(path)
        readers[cam] = reader
        if reader.is_valid:
            valid_readers.append(reader)
        else:
            print(f"⚠️ Warning: {cam} missing or invalid.")
            
    if not valid_readers:
        raise ValueError(f"No valid videos found for {uuid}")

    # Determine timing
    min_frames = min([r.frame_count for r in valid_readers])
    indices = np.linspace(0, min_frames - 1, num_frames).astype(int)
    
    # Start readers
    for r in valid_readers:
        r.start(indices)

    tiled_frames = []
    ref_w, ref_h = valid_readers[0].width, valid_readers[0].height
    
    print(f"🔄 Processing {num_frames} frames in parallel...")
    
    for _ in range(num_frames):
        # Gather frames from all queues
        current_imgs = []
        for cam in camera_list:
            reader = readers[cam]
            frame = reader.get_frame()
            
            if frame is not None:
                img = Image.fromarray(frame)
            else:
                img = Image.new('RGB', (ref_w, ref_h), (0, 0, 0))
            current_imgs.append(img)
            
        # Tile
        canvas = Image.new('RGB', (ref_w * cols, ref_h * rows), (0, 0, 0))
        for idx, img in enumerate(current_imgs):
            if idx >= cols * rows: break
            c_x, c_y = idx % cols, idx // cols
            canvas.paste(img, (c_x * ref_w, c_y * ref_h))
            
        tiled_frames.append(canvas)

    # Cleanup
    for r in readers.values():
        r.stop()
        
    return tiled_frames

def run():
    parser = argparse.ArgumentParser(description="Run inference with LoRA adapter on Multi-View Data")
    parser.add_argument("--uuid", type=str, required=True, help="Sample UUID")
    parser.add_argument("--data_dir", type=str, default="data/samples")
    parser.add_argument("--adapter_id", type=str, default=DEFAULT_ADAPTER_ID)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--track", action="store_true", help="Enable Comet ML tracking")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization for speed/VRAM")
    parser.add_argument("--project_name", type=str, default=None, help="Comet ML Project Name")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(DEFAULT_CONFIG)
    setup = cfg['vision']['camera_setup']
    camera_list = cfg['vision']['setups'][setup]['cameras']
    grid = cfg['vision']['setups'][setup]['grid']
    
    # Determine Project Name
    project_name = args.project_name if args.project_name else cfg.get('project_name', 'amd-vision-omni')

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
                "revision": args.revision or "latest",
                "quantization": "4bit" if args.load_in_4bit else "16bit"
            })
            experiment.add_tag("inference")
        else:
            print("⚠️ Comet ML requested but not installed. Skipping.")
    
    # START MONITORING
    monitor = HardwareMonitor(interval=0.5)
    monitor.start()

    try:
        print(f"⚙️  Config: {setup} ({len(camera_list)} cameras, Grid: {grid})")

        # Load Frames (Threaded)
        frames = get_tiled_frames(args.uuid, args.data_dir, camera_list, grid)
        print(f"🎞️  Prepared {len(frames)} tiled frames.")

        # Logging Input Video
        if experiment:
            temp_vid = "inference_input.mp4"
            save_video(frames, temp_vid)
            experiment.log_video(temp_vid, name=f"input_{args.uuid}", overwrite=True)

        # Load Model
        print(f"🚀 Loading Base Model: {BASE_MODEL_ID} (4-bit: {args.load_in_4bit})...")
        model, tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL_ID,
            load_in_4bit=args.load_in_4bit,
            torch_dtype=torch.float16 if args.load_in_4bit else torch.bfloat16,
        )
        
        if not args.no_lora:
            print(f"🔗 Attaching LoRA: {args.adapter_id} (Revision: {args.revision if args.revision else 'Latest'})...")
            model.load_adapter(args.adapter_id, revision=args.revision)
        
        FastVisionModel.for_inference(model) 

        # Prepare Input
        messages = [
            {"role": "system", "content": ""}, # Explicitly empty system prompt to match training
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": f"Analyze the {setup} sequence. Predict the ego-motion. Output ONLY the Displacement in meters and Velocity in m/s. Format: Displacement: <value> m, Velocity: <value> m/s"}
            ]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"DEBUG: PROMPT:\n{text}")
        inputs = tokenizer(text=[text], videos=[frames], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        # Generate
        print("🧠 Thinking (Metrics being captured directly from GPU user space)...")
        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.001) # Near-greedy decoding
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        print(f"DEBUG: Raw Token IDs: {generated_ids_trimmed}")
        
        output_text = tokenizer.batch_decode( # Changed processor to tokenizer
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"DEBUG: Raw Text Output: '{output_text}'")
        print(f"\n✅ PREDICTION:\n{output_text}") # Changed decoded to output_text

        if experiment:
            experiment.log_text(output_text, metadata={"type": "prediction"}) # Changed decoded to output_text
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        if experiment: experiment.log_text(str(e), metadata={"type": "error"})
        raise e
    finally:
        # STOP MONITORING AND LOG
        monitor.stop()
        stats = monitor.get_summary()
        
        if stats:
            print("\n📊 INFERENCE HARDWARE STATS:")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f}")
                
            if experiment:
                 experiment.log_metrics(stats)

    if experiment:
        print(f"COMET_EXPERIMENT_NAME: {experiment.get_name()}")
        if os.path.exists("inference_input.mp4"):
            os.remove("inference_input.mp4")
        experiment.end()

if __name__ == "__main__":
    run()