import os
import torch
import yaml
from datasets import load_from_disk
from unsloth import FastVisionModel, is_bfloat16_supported, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from src.training.compute import configure_compute

class MI300XVerboseLogger(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n[STEP {state.global_step}/{state.max_steps}] VRAM: {allocated:.2f}GB")

def train():
    configure_compute()
    with open("configs/finetuning_config.yaml", "r") as f: cfg = yaml.safe_load(f)
    
    # 1. Load Pre-processed Data (INSTANT)
    dataset_path = "/workspace/AMD-Vision-Omni/data/processed_dataset"
    dataset = load_from_disk(dataset_path)
    
    # 2. Model Setup
    model, tokenizer = FastVisionModel.from_pretrained(cfg['model']['base_model'], load_in_4bit=True)
    model.is_vision_model = True # Force flag
    
    model = FastVisionModel.get_peft_model(
        model, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=cfg['model']['lora_rank'], 
        lora_alpha=cfg['model']['lora_alpha']
    )

    # 3. Fast Formatting
    def format_fn(example):
        msgs = [{"role": "user", "content": [{"type": "video", "video": example["images"]}, {"type": "text", "text": example["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"<think>\n{example['reasoning']}\n</think>\nAction: {example['action']}"}]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False), "video": example["images"]}

    dataset = dataset.map(format_fn, remove_columns=dataset.column_names, num_proc=8)

    # 4. SFT Training
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = dataset,
        args = SFTConfig(
            output_dir = f"checkpoints/{cfg['project_name']}",
            per_device_train_batch_size = cfg['training']['batch_size'],
            gradient_accumulation_steps = cfg['training']['gradient_accumulation_steps'],
            max_steps = cfg['training']['max_steps'],
            learning_rate = float(cfg['training']['learning_rate']),
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = cfg['training']['optimizer'],
            dataset_text_field = "text",
        ),
        callbacks = [MI300XVerboseLogger()]
    )
    
    print("🚀 MI300X: Commencing Full-Throttle Training...")
    trainer.train()

if __name__ == "__main__":
    train()