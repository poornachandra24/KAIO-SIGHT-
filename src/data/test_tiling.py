import os
import yaml
from loader import get_dataloader

def test_tiling():
    print("--- [VERIFYING TEMPORAL TILING ARCHITECTURE] ---")
    data_cfg = "configs/data_config.yaml"
    ft_cfg = "configs/finetuning_config.yaml"
    
    try:
        loader = get_dataloader(data_cfg, ft_cfg, limit=1)
        sample = next(iter(loader))
        
        images = sample['images']
        print(f"✅ Success: Loaded {len(images)} tiled frames.")
        print(f"✅ Grid Resolution: {images[0].size}")
        
        output_file = "debug_tiling_result.jpg"
        images[0].save(output_file)
        print(f"✅ Visual check saved to: {output_file}")
        print(f"✅ Sample Action: {sample['action']}")

    except Exception as e:
        print(f"❌ [VERIFICATION FAILED]: {str(e)}")

if __name__ == "__main__":
    test_tiling()