import os
import torch

def configure_compute():
    """
    Configures environment for AMD MI300X.
    """
    # ROCm specific optimizations
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2" # Common for MI300X
    os.environ["ROCR_VISIBLE_DEVICES"] = "0" # Assume single device for now or handle multi-gpu
    
    # Flash Attention
    # Unsloth handles this internally usually, but we can force it
    # os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1" 
    
    # Memory
    # os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Compute configured for: {device_name}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: No CUDA/ROCm device found!")
