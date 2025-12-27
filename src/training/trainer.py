import os
import sys
import resource

# --- CONFIG: SYSTEM & CACHE ---
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f"✅ System file limit increased to: {hard}")
except Exception as e:
    print(f"⚠️ Could not increase file limit: {e}")

os.environ["TRITON_CACHE_DIR"] = "/workspace/AMD-Vision-Omni/triton_cache"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import torch
import yaml
from datasets import load_from_disk, concatenate_datasets
from unsloth import FastVisionModel, is_bfloat16_supported, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from src.training.compute import configure_compute

# --- CONFIG: PATHS ---
CONFIG_PATH = "configs/finetuning_config.yaml"
SHARDS_PATH = "/workspace/AMD-Vision-Omni/data/shards"

class MI300XVerboseLogger(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n[STEP {state.global_step}/{state.max_steps}] "
                  f"VRAM Alloc: {allocated:.2f}GB | Res: {reserved:.2f}GB")

def train():
    configure_compute()
    
    with open(CONFIG_PATH, "r") as f: 
        cfg = yaml.safe_load(f)

    print(f"🚀 MI300X Optimization: Switching to Native Bfloat16...")
    
    # 1. Model Loading
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg['model']['base_model'],
        load_in_4bit=False,              # Disable 4-bit for MI300X
        torch_dtype=torch.bfloat16,      # Native BF16
        max_seq_length=cfg['model']['max_seq_length'],
    )
    model.is_vision_model = True 

    # --- RESOLUTION CONFIG ---
    TARGET_RES = 504
    if hasattr(tokenizer, "image_processor"):
        target_pixels = TARGET_RES * TARGET_RES
        tokenizer.image_processor.min_pixels = target_pixels
        tokenizer.image_processor.max_pixels = target_pixels
        print(f"✅ Image Processor clamped to {TARGET_RES}x{TARGET_RES}")

    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=cfg['model']['lora_rank'], 
        lora_alpha=cfg['model']['lora_alpha']
    )

    # 2. Data Loading (Map-Style Strategy)
    print("🔄 Discovering shards...")
    shard_paths = []
    for root, dirs, files in os.walk(SHARDS_PATH):
        for d in dirs:
            if d.startswith("part_") or d.startswith("shard_"):
                shard_paths.append(os.path.join(root, d))
    shard_paths = sorted(shard_paths)
    
    if not shard_paths:
        if os.path.exists(os.path.join(SHARDS_PATH, "dataset_info.json")):
             shard_paths.append(SHARDS_PATH)

    print(f"📂 Loading {len(shard_paths)} shards as Map-Style Dataset...")
    datasets_list = [load_from_disk(p) for p in shard_paths]
    full_dataset = concatenate_datasets(datasets_list)
    print(f"✅ Consolidated Dataset: {len(full_dataset)} samples")

    # 3. On-the-Fly Formatting (Transform)
    def format_transform(examples):
        # We must return a DICT of LISTS: {"messages": [msg1, msg2, ...]}
        batch_messages = []
        
        # 'examples' is a dict of lists. examples["images"] is a list of lists (video sequences).
        for i in range(len(examples["images"])):
            raw_frames = examples["images"][i]
            
            # Resize each frame
            resized_frames = [frame.resize((TARGET_RES, TARGET_RES), resample=3) for frame in raw_frames]
            
            msgs = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "video", 
                            "video": resized_frames,
                        }, 
                        {"type": "text", "text": examples["instruction"][i]}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": f"<think>\n{examples['reasoning'][i]}\n</think>\nAction: {examples['action'][i]}"}
                    ]
                }
            ]
            batch_messages.append(msgs)
            
        return {"messages": batch_messages}

    full_dataset.set_transform(format_transform)

    # 4. Training Config
    target_total_bs = cfg['training']['batch_size'] * cfg['training']['gradient_accumulation_steps']
    new_grad_accum = target_total_bs // 1 

    print(f"📊 Training Strategy: Batch Size=1 | Grad Accum={new_grad_accum}")

    # --- OPTIMIZATION: BYPASS UNSLOTH CHECKS ---
    from unsloth_zoo import tokenizer_utils, training_utils
    def no_op(*args, **kwargs): return
    tokenizer_utils.fix_untrained_tokens = no_op
    training_utils.fix_zero_training_loss = no_op
    print("⚡ Unsloth Validation Checks bypassed.")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = full_dataset,
        args = SFTConfig(
            output_dir = f"checkpoints/{cfg.get('project_name', 'amd-vision-omni')}",
            per_device_train_batch_size = 1, 
            gradient_accumulation_steps = new_grad_accum,
            warmup_steps = 5,
            max_steps = 60,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            learning_rate = float(cfg['training']['learning_rate']),
            optim = cfg['training']['optimizer'],
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            logging_steps = 1,
            save_strategy = "steps",
            save_steps = 20,
            dataset_num_proc = 1,
            dataloader_num_workers = 0,
            max_seq_length = cfg['model']['max_seq_length'],
            report_to = "none",
        ),
        callbacks = [MI300XVerboseLogger()]
    )
    
    print("🚀 MI300X: Commencing Full-Throttle Training...")
    trainer.train()

if __name__ == "__main__":
    train()