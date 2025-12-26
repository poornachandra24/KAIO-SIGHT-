import os
import torch
import time
import yaml
from datetime import datetime
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from src.data.loader import get_dataloader
from src.training.compute import configure_compute

# --- Custom Architectural Logger Callback ---
class MI300XVerboseLogger(TrainerCallback):
    """Architectural callback to monitor ROCm telemetry and VRAM utilization."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            # Get ROCm VRAM Stats
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"\n[METRICS - STEP {state.global_step}]")
            if len(state.log_history) > 0:
                last_log = state.log_history[-1]
                print(f"  Loss: {last_log.get('loss', 'N/A'):.4f} | LR: {last_log.get('learning_rate', 'N/A'):.2e}")
            
            print(f"  MI300X VRAM: {allocated:.2f}GB / 192GB (Reserved: {reserved:.2f}GB)")
            print(f"  Utilization: {(allocated/192.0)*100:.1f}%")

def train():
    # 0. Environment Setup
    configure_compute()
    
    # 1. Load Configurations
    with open("configs/finetuning_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    MAX_SEQ_LENGTH = cfg['model']['max_seq_length']
    OUTPUT_DIR = f"checkpoints/{cfg['project_name']}-{datetime.now().strftime('%m%d_%H%M')}"
    
    print(f"--- [INITIALIZING {cfg['project_name']} ON MI300X] ---")
    
    # 2. Load Qwen3-VL-8B via Unsloth
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg['model']['base_model'],
        load_in_4bit = True,
        max_seq_length = MAX_SEQ_LENGTH,
    )
    
    # 3. Inject LoRA Adapters (Targeting Vision & Language)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = True,
        finetune_language_layers = True,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj"],
        rank = cfg['model']['lora_rank'],
        lora_alpha = cfg['model']['lora_alpha'],
        lora_dropout = 0,
    )

    # 4. Apply Qwen 2.5/3 Chat Template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")

    # 5. Multimodal "Thinking" Formatting Function
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        reasoning_blocks = examples["reasoning"] 
        actions = examples["action"]
        images_list = examples["images"] # Sequences of tiled frames

        texts = []
        for i in range(len(instructions)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": images_list[i]}, 
                        {"type": "text", "text": instructions[i]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<think>\n{reasoning_blocks[i]}\n</think>\nAction: {actions[i]}"}
                    ],
                },
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        
        return { "text": texts }

    # 6. Data Pipeline Initialization
    print(f"--- [SYNCING SPATIO-TEMPORAL DATASET] ---")
    train_loader = get_dataloader(
        "configs/data_config.yaml", 
        "configs/finetuning_config.yaml",
        limit=None # Set a number here for testing
    )
    
    # Map the formatting function to the dataset
    dataset = train_loader.dataset.map(formatting_prompts_func, batched=True)
    
    # Sanity Check
    print(f"\n[ARCHITECT CHECK] Sample Prompt Structure:")
    print(dataset[0]['text'][:600] + "...")

    # 7. Training Configuration
    training_args = SFTConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = cfg['training']['batch_size'],
        gradient_accumulation_steps = cfg['training']['gradient_accumulation_steps'],
        warmup_ratio = cfg['training']['warmup_ratio'],
        max_steps = cfg['training']['max_steps'],
        learning_rate = float(cfg['training']['learning_rate']),
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = cfg['training']['logging_steps'],
        optim = cfg['training']['optimizer'],
        weight_decay = 0.05,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "none", 
        save_steps = cfg['training']['save_steps'],
    )

    # 8. SFTTrainer Execution
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = training_args,
        callbacks = [MI300XVerboseLogger()],
    )

    # Optimize to train only on assistant responses (Thinking + Action)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    print(f"\n--- [TRAINING COMMENCED] ---")
    start_time = time.time()
    trainer.train()
    
    # 9. Model Export
    total_time = (time.time() - start_time) / 3600
    print(f"--- [TRAINING COMPLETE] ---")
    print(f"Duration: {total_time:.2f} hours. Saving to {OUTPUT_DIR}...")
    
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "lora")
    print("Export successful.")

if __name__ == "__main__":
    train()