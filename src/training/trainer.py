import os
import torch
import time
import yaml
import io
from datetime import datetime
from datasets import Dataset as HFDataset

# Unsloth must be imported before TRL/Transformers
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# Absolute imports from the project root
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
                loss = last_log.get('loss', 'N/A')
                lr = last_log.get('learning_rate', 'N/A')
                print(f"  Loss: {loss} | LR: {lr}")
            
            print(f"  MI300X VRAM: {allocated:.2f}GB / 192GB (Reserved: {reserved:.2f}GB)")
            print(f"  Utilization: {(allocated/192.0)*100:.1f}%")

def train():
    # 0. Initial Hardware Optimization
    configure_compute()
    
    # Load Configurations
    with open("configs/finetuning_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    MAX_SEQ_LENGTH = cfg['model']['max_seq_length']
    OUTPUT_DIR = f"checkpoints/{cfg['project_name']}-{datetime.now().strftime('%m%d_%H%M')}"
    
    print(f"--- [INITIALIZING {cfg['project_name']} ON MI300X] ---")
    
    # 1. Load Qwen3-VL-8B via Unsloth
    # Note: On ROCm, we typically use 4-bit to keep the vision-temporal context wide
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg['model']['base_model'],
        load_in_4bit = True,
        max_seq_length = MAX_SEQ_LENGTH,
    )
    
    # 2. Inject LoRA Adapters
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

    # 3. Apply the Qwen 2.5/3 Chat Template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")

    # 4. Multimodal "Thinking" Formatting Function
    def formatting_prompts_func(examples):
        # examples is a dict of lists provided by HF .map()
        instructions = examples["instruction"]
        reasoning_blocks = examples["reasoning"] 
        actions = examples["action"]
        images_sequences = examples["images"] # List of lists of PIL images

        texts = []
        for i in range(len(instructions)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": images_sequences[i]}, 
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

    # 5. Data Pipeline - The "Architect's Bridge"
    print(f"--- [SYNCING SPATIO-TEMPORAL DATASET] ---")
    
    # get_dataloader returns our custom NuRecTemporalDataset wrapped in a DataLoader
    train_loader = get_dataloader(
        "configs/data_config.yaml", 
        "configs/finetuning_config.yaml",
        limit=None 
    )
    
    py_dataset = train_loader.dataset

    # Bridge: Convert PyTorch Dataset to HuggingFace Dataset via generator
    def gen():
        for i in range(len(py_dataset)):
            yield py_dataset[i]

    hf_dataset = HFDataset.from_generator(gen)
    
    # Now we can use the HF .map() functionality
    dataset = hf_dataset.map(formatting_prompts_func, batched=True)
    
    # Sanity Check
    print(f"\n[ARCHITECT CHECK] Sample Prompt Structure (First 500 chars):")
    print(dataset[0]['text'][:500] + "...")

    # 6. Training Configuration
    training_args = SFTConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = cfg['training']['batch_size'],
        gradient_accumulation_steps = cfg['training']['gradient_accumulation_steps'],
        warmup_ratio = cfg['training']['warmup_ratio'],
        max_steps = cfg['training']['max_steps'],
        learning_rate = float(cfg['training']['learning_rate']),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = cfg['training']['logging_steps'],
        optim = cfg['training']['optimizer'],
        weight_decay = 0.05,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "none", 
        save_steps = cfg['training']['save_steps'],
    )

    # 7. SFTTrainer Initialization
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = training_args,
        callbacks = [MI300XVerboseLogger()],
    )

    # 8. Train on Responses Only
    # This ensures the model learns to output the <think> and Action blocks
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    print(f"\n--- [TRAINING COMMENCED ON AMD INSTINCT MI300X] ---")
    start_time = time.time()
    
    trainer.train()
    
    total_time = (time.time() - start_time) / 3600
    print(f"--- [TRAINING COMPLETE] ---")
    print(f"Total Duration: {total_time:.2f} hours. Saving model to {OUTPUT_DIR}...")
    
    # 9. Final Save (LoRA Adapters)
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "lora")
    print("Export successful. Ready for Inference.")

if __name__ == "__main__":
    train()