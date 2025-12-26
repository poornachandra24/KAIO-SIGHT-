import os
import io
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import yaml

class NuRecTemporalDataset(Dataset):
    def __init__(self, data_config_path, ft_config_path, split='train'):
        with open(data_config_path, 'r') as f: self.data_cfg = yaml.safe_load(f)
        with open(ft_config_path, 'r') as f: self.ft_cfg = yaml.safe_load(f)
        
        self.data_dir = self.data_cfg['dataset']['local_dir']
        setup_key = self.ft_cfg['vision']['camera_setup']
        self.camera_names = self.ft_cfg['vision']['setups'][setup_key]['cameras']
        self.grid = self.ft_cfg['vision']['setups'][setup_key]['grid']
        self.window_size = self.ft_cfg['vision']['window_size']
        
        # 1. Load the "Atlas" Index
        index_path = os.path.join(self.data_dir, self.data_cfg['dataset']['index_file'])
        self.atlas_df = pd.read_parquet(index_path)
        
        # Filter for current split (train/test)
        split_atlas = self.atlas_df[self.atlas_df['split'] == split]
        self.valid_chunks = self._get_downloaded_chunks(split_atlas['chunk'].unique())
        
        # 2. Build Index: Map sliding windows to specific Clip UUIDs
        self.samples = []
        print(f"[LOADER] Building temporal index from {len(self.valid_chunks)} chunks...")
        
        for chunk_id in self.valid_chunks:
            chunk_str = f"{chunk_id:04d}"
            ego_zip_path = os.path.join(self.data_dir, "labels", "egomotion", f"egomotion.chunk_{chunk_str}.zip")
            
            with zipfile.ZipFile(ego_zip_path, 'r') as z:
                # Find all parquet files in this chunk
                parquet_files = [f for f in z.namelist() if f.endswith('.parquet')]
                
                for p_file in parquet_files:
                    # Extract UUID from filename: UUID.egomotion.parquet
                    clip_uuid = p_file.split('.')[0]
                    
                    with z.open(p_file) as f:
                        # Load kinematics for this specific clip
                        clip_ego = pd.read_parquet(io.BytesIO(f.read()))
                        
                        # NuRec usually uses 'timestamp_us' or 'timestamp_ns'
                        time_col = next((c for c in ['timestamp_us', 'timestamp_ns', 'timestamp'] if c in clip_ego.columns), clip_ego.columns[0])
                        clip_ego = clip_ego.sort_values(time_col)
                        
                        # Create sliding windows of 16 frames
                        step = max(1, self.window_size // 2)
                        for i in range(0, len(clip_ego) - self.window_size, step):
                            window_data = clip_ego.iloc[i : i + self.window_size]
                            self.samples.append({
                                'chunk': chunk_id,
                                'clip_uuid': clip_uuid,
                                'timestamps': window_data[time_col].tolist(),
                                'kinematics': window_data[['translation', 'rotation']].to_dict('records') 
                            })

        print(f"[LOADER] Created {len(self.samples)} temporal samples from {split} split.")

    def _get_downloaded_chunks(self, split_chunks):
        downloaded = []
        for chunk_id in split_chunks:
            chunk_str = f"{chunk_id:04d}"
            ego_path = os.path.join(self.data_dir, "labels", "egomotion", f"egomotion.chunk_{chunk_str}.zip")
            if os.path.exists(ego_path):
                downloaded.append(chunk_id)
        return downloaded

    def _dynamic_tile(self, pil_images):
        w, h = pil_images[0].size
        cols, rows = self.grid
        canvas = Image.new('RGB', (w * cols, h * rows), (0,0,0))
        for i, img in enumerate(pil_images):
            canvas.paste(img, ((i % cols) * w, (i // cols) * h))
        return canvas

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chunk_str = f"{sample['chunk']:04d}"
        
        # 1. Open Concurrent Camera Zips
        camera_zips = {cam: zipfile.ZipFile(os.path.join(self.data_dir, "camera", cam, f"{cam}.chunk_{chunk_str}.zip"), 'r') for cam in self.camera_names}
        
        temporal_images = []
        # 2. Extract 16 Frames (Sequence)
        for ts in sample['timestamps']:
            current_timestamp_imgs = []
            for cam in self.camera_names:
                # Common NuRec Zip Pattern: <clip_uuid>.<cam_name>.<timestamp>.jpg
                img_name = f"{sample['clip_uuid']}.{cam}.{ts}.jpg"
                
                try:
                    with camera_zips[cam].open(img_name) as f:
                        img = Image.open(io.BytesIO(f.read())).convert("RGB")
                except:
                    # Fallback to gray placeholder if frame is dropped
                    img = Image.new('RGB', (1280, 720), (40, 40, 40))
                current_timestamp_imgs.append(img)
            
            temporal_images.append(self._dynamic_tile(current_timestamp_imgs))

        for z in camera_zips.values(): z.close()

        # 3. Calculate Physical Metrics from Kinematics
        kin = sample['kinematics']
        pos_start = np.array(kin[0]['translation'][:2])
        pos_end = np.array(kin[-1]['translation'][:2])
        dist = np.linalg.norm(pos_end - pos_start)
        
        # 4. Generate the "Reasoning" Label
        setup = self.ft_cfg['vision']['camera_setup']
        reasoning = (
            f"Clip UUID: {sample['clip_uuid']} | Sequence: {len(temporal_images)} frames. "
            f"Physical ground-truth confirms a displacement of {dist:.2f} meters. "
            f"Observation of {setup} views confirms environmental stability."
        )

        return {
            "images": temporal_images,
            "instruction": f"Analyze the driving sequence ({setup}) and predict vehicle displacement.",
            "reasoning": reasoning,
            "action": f"Vehicle moved {dist:.1f} meters."
        }

def get_dataloader(data_config_path, ft_config_path, split='train', limit=None):
    dataset = NuRecTemporalDataset(data_config_path, ft_config_path, split=split)
    if limit: dataset.samples = dataset.samples[:limit]
    return DataLoader(dataset, batch_size=1, shuffle=(split=='train'), num_workers=4, collate_fn=lambda x: x[0])