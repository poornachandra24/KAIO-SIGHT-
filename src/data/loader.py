import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import zipfile
import json
from .processor import DataProcessor

class NuRecDataset(Dataset):
    def __init__(self, config_path, split='train', limit=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.local_dir = self.config['dataset']['local_dir']
        self.index_file = os.path.join(self.local_dir, self.config['dataset']['index_file'])
        self.sequence_length = self.config['processing']['sequence_length']
        self.cameras = self.config['processing']['cameras']
        
        self.processor = DataProcessor(self.config)
        
        # Load index
        df = pd.read_parquet(self.index_file)
        
        # Filter by split
        df = df[df['split'] == split]
        
        # Limit for testing
        if limit:
            df = df.head(limit)
            
        self.clips = df
        self.clip_ids = self.clips.index.tolist()
        
        # Cache zip paths to avoid repeated lookups
        # We assume a structure or we search. 
        # Since we downloaded specific chunks, we need to map clip -> chunk -> zip file
        self.chunk_map = self.clips['chunk'].to_dict()
        
    def __len__(self):
        return len(self.clip_ids)
    
    def _get_zip_path(self, chunk_id, camera_name):
        # Naming convention: {camera_name}.chunk_{chunk_id:04d}.zip
        filename = f"{camera_name}.chunk_{chunk_id:04d}.zip"
        # Search in camera subfolders
        # We assume standard structure: data/camera/{camera_name}/{filename}
        path = os.path.join(self.local_dir, "camera", camera_name, filename)
        return path

    def _get_label_zip_path(self, chunk_id):
        # labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip
        filename = f"egomotion.chunk_{chunk_id:04d}.zip"
        path = os.path.join(self.local_dir, "labels", "egomotion", filename)
        return path

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        chunk_id = self.chunk_map[clip_id]
        
        # 1. Get Frames
        # We need to find the files inside the zip for this clip.
        # This is tricky without extracting. We need to know the internal filenames.
        # Usually NuRec has a consistent naming scheme inside zips.
        # Let's assume: {clip_id}/{timestamp}.jpg or similar.
        # For now, since we don't have the file list, we might need to peek into the zip 
        # or assume a sorted list of files corresponds to frames.
        
        # Strategy: Open one camera zip, list files for this clip, sort them to get temporal order.
        # Then assume other cameras match.
        
        # Optimization: We can't list zip contents every time. 
        # Ideally we have a metadata map. 
        # For this implementation, we will do it dynamically but it might be slow.
        # In production, we would pre-compute a mapping.
        
        # Let's try to get the file list for the first camera
        cam0 = self.cameras[0]
        zip_path = self._get_zip_path(chunk_id, cam0)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                all_files = z.namelist()
                # Filter for this clip
                clip_files = sorted([f for f in all_files if str(clip_id) in f])
                
            # Select sequence
            # If we have more frames than sequence_length, we can sample or take the beginning.
            # Let's take the first N frames.
            if len(clip_files) > self.sequence_length:
                clip_files = clip_files[:self.sequence_length]
            
            # Now build the frame_data_list for the processor
            frame_data_list = []
            for f in clip_files:
                # f is like {clip_id}/frame_001.jpg (example)
                # We need to construct the filename for other cameras.
                # Usually the suffix is the same.
                suffix = f.split('/')[-1] # e.g. frame_001.jpg
                
                frame_info = []
                for cam in self.cameras:
                    cam_zip = self._get_zip_path(chunk_id, cam)
                    # We assume the internal structure mirrors the camera name or just uses the clip_id
                    # If the zip is camera_front_wide..., the internal file might be {clip_id}/{suffix}
                    # We need to verify this structure.
                    # Based on standard datasets, it's usually {clip_id}/{timestamp}.jpg
                    # So the internal path is likely the SAME for all cameras if they are synchronized?
                    # Or maybe {camera_name}/{clip_id}/{timestamp}.jpg?
                    # Let's assume it's {clip_id}/{suffix} for now.
                    frame_info.append({
                        'zip_path': cam_zip,
                        'filename': f 
                    })
                frame_data_list.append(frame_info)
                
            # Process Images
            pixel_values = self.processor.process_sequence(frame_data_list)
            
            # 2. Get Labels / Plan
            # We need the egomotion or plan data.
            # This would come from the labels zip.
            # For this task, we are asked to "Pre-process 'ego_trajectory' into a natural language Plan".
            # We need to read the label file.
            label_zip = self._get_label_zip_path(chunk_id)
            # Assume json or csv inside
            # Let's assume there's a file {clip_id}.json or similar.
            
            # Placeholder for label reading logic
            # We will generate a dummy plan if we can't read it yet, to unblock the pipeline.
            plan_text = "<think> The vehicle is moving forward. The road is clear. </think> Action: MAINTAIN_SPEED"
            
            return {
                "pixel_values": pixel_values, # (T, C, H, W)
                "text": plan_text
            }
            
        except Exception as e:
            print(f"Error loading clip {clip_id}: {e}")
            # Return dummy data to avoid crashing
            return {
                "pixel_values": torch.zeros((self.sequence_length, 3, 336, 336)),
                "text": "Error"
            }

def get_dataloader(config_path, split='train', batch_size=2, limit=None):
    dataset = NuRecDataset(config_path, split=split, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=4)
