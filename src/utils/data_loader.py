import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Tuple

class CausalImageDataset(Dataset):
    """
    Standard dataset for loading CausalVAE/LANCA data (Pendulum, Flow, Candle).
    Expects data to be in: root_dir/dataset_name/train_metadata.csv
    """
    def __init__(self, root_dir, dataset_name, train=True, transform=None):
        self.dataset_name = dataset_name
        self.transform = transform
        split = 'train' if train else 'test'
        
        # Path handling for standard vs CANDLE structure
        if 'candle' in dataset_name.lower():
            self.data_dir = os.path.join(root_dir, 'candle') # CSVs here
            self.image_dir = os.path.join(root_dir, 'IITH_CANDLE') # Images here
        else:
            self.data_dir = os.path.join(root_dir, dataset_name)
            self.image_dir = os.path.join(self.data_dir, split)

        metadata_path = os.path.join(self.data_dir, f"{split}_metadata.csv")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata CSV not found at: {metadata_path}")
            
        self.metadata = pd.read_csv(metadata_path)
        self.image_files = self.metadata['filename'].values

        # Detect Bounding Boxes
        self.has_bbox = 'min_x' in self.metadata.columns
        self.orig_w, self.orig_h = 320, 240 # Standard CANDLE resolution

        # Identify label columns dynamically or hardcoded based on known datasets
        if 'pendulum' in dataset_name:
            self.label_cols = ['i', 'j', 'shade', 'mid']
        elif 'flow' in dataset_name:
            self.label_cols = ['r', 'h', 'x_true_10x', 'hole']
        elif 'candle' in dataset_name:
            self.label_cols = ['light', 'scene', 'object', 'color', 'size', 'rotation']
        else:
            # Fallback: try to detect numeric columns excluding filename/bbox columns
            exclude = ['filename', 'min_x', 'min_y', 'max_x', 'max_y']
            self.label_cols = [c for c in self.metadata.columns if c not in exclude and '_norm' not in c]

        # Convert labels to tensor
        self.labels = torch.tensor(self.metadata[self.label_cols].values, dtype=torch.float32)
        
        # Normalization Stats (Compute or Placeholder)
        self.label_mean = torch.mean(self.labels, dim=0)
        self.label_std = torch.std(self.labels, dim=0)
        # Avoid division by zero
        self.label_std = torch.nan_to_num(self.label_std, nan=1.0)
        self.label_std[self.label_std == 0] = 1.0

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 96, 96), torch.zeros_like(self.label_mean), torch.zeros(1, 96, 96)

        if self.transform:
            image = self.transform(image)
            
        # Get Label & Normalize
        label_raw = self.labels[idx]
        label_norm = (label_raw - self.label_mean) / self.label_std
        
        # --- DYNAMIC RESIZING MASK GENERATION ---
        # Get dimensions from the transformed tensor
        target_h, target_w = image.shape[1], image.shape[2]
        mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)

        if self.has_bbox:
            row = self.metadata.iloc[idx]
            
            # Calculate Scale factors (Target / Original)
            scale_x = target_w / self.orig_w
            scale_y = target_h / self.orig_h
            
            # Scale X (Width is standard)
            sx1 = int(np.clip(row['min_x'] * scale_x, 0, target_w))
            sx2 = int(np.clip(row['max_x'] * scale_x, 0, target_w))
            
            # Scale Y (Invert for Bottom-Left Origin)
            # In metadata: min_y is distance from bottom. 
            # In Tensor image: index 0 is top.
            # Top edge of box = (Height - max_y_scaled)
            # Bottom edge of box = (Height - min_y_scaled)
            
            sy1_raw = target_h - (row['max_y'] * scale_y)
            sy2_raw = target_h - (row['min_y'] * scale_y)
            
            sy1 = int(np.clip(sy1_raw, 0, target_h))
            sy2 = int(np.clip(sy2_raw, 0, target_h))
            
            # Fill mask (sy1 is top edge, sy2 is bottom edge, so sy1 < sy2)
            mask[:, sy1:sy2, sx1:sx2] = 1.0

        return image, label_norm, mask

    def set_label_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Align validation normalization with training stats.
        """
        self.label_mean = mean
        self.label_std = std