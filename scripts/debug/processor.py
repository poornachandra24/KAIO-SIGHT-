import zipfile
import io
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.image_size = (336, 336) # Standard Qwen-VL size, or configurable
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def read_image_from_zip(self, zip_path, filename):
        """
        Reads an image directly from a zip file into memory.
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(filename) as f:
                    image_data = f.read()
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    return image
        except Exception as e:
            print(f"Error reading {filename} from {zip_path}: {e}")
            return None

    def tile_images(self, images):
        """
        Tiles 4 images into a 2x2 grid.
        images: List of 4 PIL Images [Front, Rear, Left, Right] (or user defined order)
        Returns: Single PIL Image (2x2 grid)
        """
        if len(images) != 4:
            raise ValueError("Expected 4 images for tiling")

        w, h = images[0].size
        grid_w, grid_h = w * 2, h * 2
        grid_image = Image.new('RGB', (grid_w, grid_h))

        # Layout:
        # Front | Right
        # ------+------
        # Left  | Rear
        # This layout is arbitrary, need to confirm preferred ego-centric layout.
        # Let's assume:
        # Top-Left: Front
        # Top-Right: Right
        # Bottom-Left: Left
        # Bottom-Right: Rear
        
        grid_image.paste(images[0], (0, 0))       # Front
        grid_image.paste(images[1], (w, 0))       # Right
        grid_image.paste(images[2], (0, h))       # Left
        grid_image.paste(images[3], (w, h))       # Rear
        
        return grid_image

    def process_sequence(self, frame_data_list):
        """
        Process a sequence of frames.
        frame_data_list: List of dicts, each containing paths/filenames for 4 cameras.
        """
        processed_frames = []
        for frame_data in frame_data_list:
            images = []
            # Order matters here. 
            # config['processing']['cameras'] should define the order.
            # We need to map the config camera names to the actual files in the zip.
            
            # This part requires the Loader to pass the correct zip path and internal filename.
            # The processor just does the heavy lifting of IO and Tiling.
            
            for cam_info in frame_data:
                img = self.read_image_from_zip(cam_info['zip_path'], cam_info['filename'])
                if img:
                    images.append(img)
                else:
                    # Handle missing frame? Black frame?
                    images.append(Image.new('RGB', self.image_size, (0, 0, 0)))
            
            tiled_img = self.tile_images(images)
            processed_frames.append(self.transform(tiled_img))
            
        return torch.stack(processed_frames)
