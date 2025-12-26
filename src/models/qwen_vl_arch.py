from unsloth import FastVisionModel
import torch

def get_model_and_tokenizer(model_name="unsloth/Qwen2-VL-7B-Instruct", max_seq_length=65536):
    """
    Initializes the Qwen2-VL model (proxy for Qwen3) using Unsloth.
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False, # Use 16-bit for MI300X if possible, or 4bit for efficiency
        use_gradient_checkpointing="unsloth",
    )
    
    # Enable PEFT / LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, 
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer
