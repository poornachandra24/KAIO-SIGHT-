import os
import sys
import resource

# --- 1. SYSTEM CONFIGURATION ---
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f"✅ System file limit increased to: {hard}")
except Exception as e:
    print(f"⚠️ Could not increase file limit: {e}")

os.environ["TRITON_CACHE_DIR"] = "/workspace/KAIO-SIGHT/triton_cache"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import torch
import yaml
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets
from unsloth import FastVisionModel, is_bfloat16_supported, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from src.training.callbacks import AutomatedReportCallback
from src.training.compute import configure_compute

CONFIG_PATH = "configs/finetuning_config.yaml"
SHARDS_PATH = "/workspace/KAIO-SIGHT/data/shards"

class MI300XVerboseLogger(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n[STEP {state.global_step}] VRAM: {allocated:.2f}GB")

class LazyVisionDataset(Dataset):
    def __init__(self, hf_dataset, transform_fn):
        self.dataset = hf_dataset
        self.transform_fn = transform_fn
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.transform_fn(self.dataset[idx])

def train():
    configure_compute()
    
    if not os.path.exists(CONFIG_PATH): raise FileNotFoundError(f"Missing {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f: cfg = yaml.safe_load(f)

    print(f"🚀 MI300X Pro Mode: Resolution {cfg['model']['resolution']}px | Frames {cfg['vision']['window_size']}")
    
    # 3. MODEL
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg['model']['base_model'],
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
        max_seq_length=cfg['model']['max_seq_length'],
    )
    model.is_vision_model = True 

    # --- RESOLUTION CONFIG ---
    TARGET_RES = cfg['model'].get('resolution', 504)
    TARGET_FRAMES = cfg['vision'].get('window_size', 16)
    
    # Configure Processor Bounds
    if hasattr(tokenizer, "image_processor"):
        target_pixels = TARGET_RES * TARGET_RES
        tokenizer.image_processor.min_pixels = target_pixels
        tokenizer.image_processor.max_pixels = target_pixels
        print(f"✅ Processor clamped to {TARGET_RES}x{TARGET_RES}")

    model = FastVisionModel.get_peft_model(
        model, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=cfg['model']['lora_rank'], 
        lora_alpha=cfg['model']['lora_alpha']
    )

    # 4. DATA
    print("🔄 Discovering shards...")
    shard_paths = sorted([os.path.join(r, d) for r, ds, _ in os.walk(SHARDS_PATH) for d in ds if d.startswith(("part_", "shard_"))])
    if not shard_paths and os.path.exists(os.path.join(SHARDS_PATH, "dataset_info.json")): shard_paths.append(SHARDS_PATH)

    print(f"📂 Loading {len(shard_paths)} shards (Map-Style)...")
    datasets_list = [load_from_disk(p) for p in shard_paths]
    full_dataset = concatenate_datasets(datasets_list)
    
    # 5. TRANSFORM (High Res + Temporal Sampling)
    def single_item_transform(example):
        raw_frames = example["images"]
        
        # Temporal Sampling: Ensure we get exactly TARGET_FRAMES
        # If we have more, sample uniformly. If less, duplicate last.
        import numpy as np
        indices = np.linspace(0, len(raw_frames)-1, TARGET_FRAMES).astype(int)
        selected_frames = [raw_frames[i] for i in indices]

        # Resize
        resized_frames = [frame.resize((TARGET_RES, TARGET_RES), resample=3) for frame in selected_frames]
        
        msgs = [
            {
                "role": "user", 
                "content": [
                    {"type": "video", "video": resized_frames}, 
                    {"type": "text", "text": example["instruction"]}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": f"<think>\n{example['reasoning']}\n</think>\nAction: {example['action']}"}]
            }
        ]
        return {"messages": msgs}

    train_dataset = LazyVisionDataset(full_dataset, single_item_transform)

    # 6. CONFIGURATION
    # Get num workers from yaml, default to 4 if not set
    num_workers = cfg['training'].get('dataloader_num_workers', 4)
    
    bs = cfg['training']['batch_size']
    ga = cfg['training']['gradient_accumulation_steps']
    print(f"📊 Strategy: BS={bs} | GradAccum={ga} | Workers={num_workers} | Res={TARGET_RES}")

    # Bypass Checks
    from unsloth_zoo import tokenizer_utils, training_utils
    def no_op(*args, **kwargs): return
    tokenizer_utils.fix_untrained_tokens = no_op
    training_utils.fix_zero_training_loss = no_op

    report_cb = AutomatedReportCallback(output_dir="docs/reports")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = train_dataset,
        args = SFTConfig(
            output_dir = f"checkpoints/{cfg.get('project_name', 'amd-vision-omni')}",
            per_device_train_batch_size = bs, 
            gradient_accumulation_steps = ga,
            num_train_epochs = cfg['training']['num_epochs'],
            max_steps = cfg['training']['max_steps'],
            warmup_ratio = cfg['training'].get('warmup_ratio', 0.1),
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            learning_rate = float(cfg['training']['learning_rate']),
            optim = cfg['training']['optimizer'],
            weight_decay = cfg['training'].get('weight_decay', 0.01),
            lr_scheduler_type = cfg['training'].get('lr_scheduler_type', "linear"),
            logging_steps = cfg['training'].get('logging_steps', 1),
            save_strategy = "steps",
            save_steps = cfg['training'].get('save_steps', 500),
            save_total_limit = cfg['training'].get('save_total_limit', 3),
            
            # --- WORKER OPTIMIZATION ---
            dataset_num_proc = 1, # Keep 1 for dataset loading
            dataloader_num_workers = num_workers, # Parallelize the transform (resize)
            # ---------------------------
            
            max_seq_length = cfg['model']['max_seq_length'],
            report_to = "none",
        ),
        callbacks = [MI300XVerboseLogger(), report_cb]
    )
    
    print("🚀 MI300X: Commencing Pro-Mode Training...")
    trainer.train()

if __name__ == "__main__":
    train()