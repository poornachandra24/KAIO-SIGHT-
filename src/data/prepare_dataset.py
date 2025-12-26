import os
import yaml
from datasets import Dataset as HFDataset
from src.data.loader import NuRecTemporalDataset

def prepare():
    # Load configs
    data_cfg = "configs/data_config.yaml"
    ft_config = "configs/finetuning_config.yaml"
    
    print("--- [ETL PHASE: PARALLEL VIDEO EXTRACTION] ---")
    # Initialize the dataset logic
    base_ds = NuRecTemporalDataset(data_cfg, ft_config, split='train')
    
    def gen():
        # This generator will be called by multiple processes
        for i in range(len(base_ds)):
            yield base_ds[i]

    # Convert to HF Dataset with 16 parallel workers
    # This turns 7 seconds per sample into ~0.4 seconds per sample effectively
    print(f"Slicing and Tiling {len(base_ds)} samples across 16 CPU cores...")
    hf_ds = HFDataset.from_generator(gen, num_proc=16)

    # Save to the 5TB Scratch Disk
    output_path = "/workspace/AMD-Vision-Omni/data/processed_dataset"
    print(f"Writing Arrow Binary format to {output_path}...")
    hf_ds.save_to_disk(output_path)
    
    print(f"✅ ETL COMPLETE. Disk usage: {os.popen(f'du -sh {output_path}').read()}")

if __name__ == "__main__":
    prepare()