import os
import torch
import yaml
from datetime import datetime
from datasets import load_from_disk, IterableDataset

# Unsloth must be imported before TRL/Transformers
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from src.data.loader import NuRecTemporalDataset
from src.training.compute import configure_compute

# --- Custom Architectural Logger ---
class MI300XVerboseLogger(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n[STEP {state.global_step}]")
            if len(state.log_history) > 0:
                last_log = state.log_history[-1]
                print(f"  Loss: {last_log.get('loss', 'N/A'):.4f}")
            print(f"  MI300X VRAM: {allocated:.2f}GB / 192GB")

def train():
    configure_compute()
    
    with open("configs/finetuning_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    PROCESSED_PATH = "/workspace/AMD-Vision-Omni/data/processed_dataset"
    USE_PROCESSED = os.getenv("USE_PROCESSED_DATA") == "1"
    MAX_SEQ_LENGTH = cfg['model']['max_seq_length']
    
    print(f"--- [INITIALIZING PROJECT OMNI: QWEN3-VL MULTIMODAL STACK] ---")

    # 1. Load Model & Tokenizer
    # ARCHITECT'S NOTE: We do NOT apply get_chat_template here. 
    # Qwen3-VL loads its native multimodal template by default.
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg['model']['base_model'],
        load_in_4bit = True,
        max_seq_length = MAX_SEQ_LENGTH,
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = True,
        finetune_language_layers = True,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        rank = cfg['model']['lora_rank'],
        lora_alpha = cfg['model']['lora_alpha'],
    )

    # 2. Data Selection Logic
    if USE_PROCESSED and os.path.exists(PROCESSED_PATH):
        print("🚀 MODE: LIGHT-SPEED (Loading Pre-processed Binary Cache)")
        dataset = load_from_disk(PROCESSED_PATH)
        is_streaming = False
    else:
        print("⚠️ MODE: JIT-STREAMING (Background Video Decoding Active)")
        raw_py_dataset = NuRecTemporalDataset("configs/data_config.yaml", "configs/finetuning_config.yaml")
        dataset = IterableDataset.from_generator(lambda: (raw_py_dataset[i] for i in range(len(raw_py_dataset))))
        is_streaming = True

    # 3. Multimodal Formatting Function (Updated for Qwen3-VL Native Template)
    def format_fn(example):
        # Qwen3-VL requires specific keys for the chat template to handle the video list
        if isinstance(example.get("instruction"), list):
            # Batched processing (Cached mode)
            texts = []
            for i in range(len(example["instruction"])):
                msgs = [
                    {"role": "user", "content": [
                        {"type": "video", "video": example["images"][i]}, 
                        {"type": "text", "text": example["instruction"][i]}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": f"<think>\n{example['reasoning'][i]}\n</think>\nAction: {example['action'][i]}"}
                    ]}
                ]
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            return {"text": texts}
        else:
            # Single item processing (Streaming mode)
            msgs = [
                {"role": "user", "content": [
                    {"type": "video", "video": example["images"]}, 
                    {"type": "text", "text": example["instruction"]}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"<think>\n{example['reasoning']}\n</think>\nAction: {example['action']}"}
                ]}
            ]
            return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    # Apply formatting on-the-fly
    dataset = dataset.map(format_fn, batched=(not is_streaming))

    # 4. Training Arguments Optimized for MI300X
    training_args = SFTConfig(
        output_dir = f"checkpoints/{cfg['project_name']}",
        per_device_train_batch_size = cfg['training']['batch_size'],
        gradient_accumulation_steps = cfg['training']['gradient_accumulation_steps'],
        max_steps = cfg['training']['max_steps'],
        learning_rate = float(cfg['training']['learning_rate']),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = cfg['training']['optimizer'],
        report_to = "none",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_text_field = "text",
        # Avoid race conditions in streaming mode
        dataloader_num_workers = 0 if is_streaming else 4,
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = training_args,
        callbacks = [MI300XVerboseLogger()],
    )

    # Architect's Strategy: Focus only on Assistant responses to learn "Thinking"
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    print("\n--- [TRAINING COMMENCED ON MI300X] ---")
    trainer.train()
    
    # Save the physical reasoning LoRA adapters
    model.save_pretrained_merged(training_args.output_dir, tokenizer, save_method = "lora")

if __name__ == "__main__":
    train()