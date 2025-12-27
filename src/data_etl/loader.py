import os
import io
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import yaml
import cv2

class NuRecTemporalDataset(Dataset):
    def __init__(self, data_config_path, ft_config_path, split='train'):
        with open(data_config_path, 'r') as f: self.data_cfg = yaml.safe_load(f)
        with open(ft_config_path, 'r') as f: self.ft_cfg = yaml.safe_load(f)
        self.data_dir = self.data_cfg['dataset']['local_dir']
        setup_key = self.ft_cfg['vision']['camera_setup']
        self.camera_names = self.ft_cfg['vision']['setups'][setup_key]['cameras']
        self.grid = self.ft_cfg['vision']['setups'][setup_key]['grid']
        self.window_size = self.ft_cfg['vision']['window_size']
        
        index_path = os.path.join(self.data_dir, self.data_cfg['dataset']['index_file'])
        self.atlas_df = pd.read_parquet(index_path)
        split_chunks = self.atlas_df[self.atlas_df['split'] == split]['chunk'].unique()
        self.valid_chunks = self._get_downloaded_chunks(split_chunks)
        
        self.samples = []
        for chunk_id in self.valid_chunks:
            chunk_str = f"{chunk_id:04d}"
            ego_zip_path = os.path.join(self.data_dir, "labels", "egomotion", f"egomotion.chunk_{chunk_str}.zip")
            with zipfile.ZipFile(ego_zip_path, 'r') as z:
                for p_file in [f for f in z.namelist() if f.endswith('.parquet')]:
                    clip_uuid = p_file.split('.')[0]
                    with z.open(p_file) as f:
                        df = pd.read_parquet(io.BytesIO(f.read())).sort_values('timestamp')
                        step = max(1, self.window_size // 2)
                        for i in range(0, len(df) - self.window_size, step):
                            window = df.iloc[i : i + self.window_size]
                            self.samples.append({
                                'chunk': chunk_id, 'clip_uuid': clip_uuid,
                                'start_frame': i, 'pos': window[['x', 'y', 'z']].values.tolist(),
                                'vel': window[['vx', 'vy']].values.tolist()
                            })

    def _get_downloaded_chunks(self, split_chunks):
        downloaded = []
        for c in split_chunks:
            c_str = f"{c:04d}"
            if os.path.exists(os.path.join(self.data_dir, "labels", "egomotion", f"egomotion.chunk_{c_str}.zip")):
                downloaded.append(c)
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
        pid = os.getpid() # Architect's Fix: Unique ID for this process
        
        temp_paths = {}
        for cam in self.camera_names:
            zip_p = os.path.join(self.data_dir, "camera", cam, f"{cam}.chunk_{chunk_str}.zip")
            target_mp4 = f"{sample['clip_uuid']}.{cam}.mp4"
            # Unique temp path per worker
            target_path = f"/dev/shm/{pid}_{target_mp4}"
            
            if not os.path.exists(target_path):
                with zipfile.ZipFile(zip_p, 'r') as z:
                    if target_mp4 in z.namelist():
                        z.extract(target_mp4, "/dev/shm")
                        os.rename(f"/dev/shm/{target_mp4}", target_path)
            temp_paths[cam] = target_path

        caps = {cam: cv2.VideoCapture(p) for cam, p in temp_paths.items()}
        temporal_images = []
        for f_offset in range(self.window_size):
            current_timestamp_imgs = []
            for cam in self.camera_names:
                cap = caps.get(cam)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_frame'] + f_offset)
                    ret, frame = cap.read()
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else Image.new('RGB', (1280, 720), (30,30,30))
                else:
                    img = Image.new('RGB', (1280, 720), (0,0,0))
                current_timestamp_imgs.append(img)
            temporal_images.append(self._dynamic_tile(current_timestamp_imgs))

        for cap in caps.values(): cap.release()
        for p in temp_paths.values(): 
            try: os.remove(p) 
            except: pass

        p_start, p_end = np.array(sample['pos'][0]), np.array(sample['pos'][-1])
        dist = np.linalg.norm(p_end[:2] - p_start[:2])
        v_avg = np.mean([np.linalg.norm(v) for v in sample['vel']])

        return {
            "images": temporal_images,
            "instruction": "Evaluate the 4-view temporal sequence and predict displacement.",
            "reasoning": f"Avg Velocity: {v_avg:.2f}m/s. Total displacement: {dist:.2f}m.",
            "action": f"Displacement of {dist:.1f} meters."
        }

def get_dataloader(data_cfg_path, ft_config_path, split='train', limit=None, batch_size=None, num_workers=0):
    dataset = NuRecTemporalDataset(data_cfg_path, ft_config_path, split=split)
    if limit:
        dataset.samples = dataset.samples[:limit]
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)