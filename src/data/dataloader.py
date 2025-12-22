import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocessor import preprocess_for_training
from src.data.normalization import S2SNormalizer

class S2SDataset(Dataset):
    """
    S2S Forecasting Dataset - Real Data with Normalization
    """
    def __init__(self, data_dir, years, ethiopia_bbox=None, mode='train', normalizer=None, lead_weeks=1):
        """
        Args:
            data_dir (str): Path to data/raw directory
            years (list): Years to load (e.g., [2000, 2001])
            ethiopia_bbox (dict): Ethiopia bounding box
            mode (str): 'train', 'val', or 'test'
            normalizer (S2SNormalizer): Pre-fitted normalizer (for val/test) or None (for train)
            lead_weeks (int): Forecast lead time in weeks
        """
        self.mode = mode
        self.input_vars = ['z', 'q', 'u', 'v', 'sst']
        self.target_vars = ['precip']
        self.lead_weeks = lead_weeks
        
        # Default Ethiopia bbox
        if ethiopia_bbox is None:
            ethiopia_bbox = {
                'lat_min': 3,
                'lat_max': 15,
                'lon_min': 33,
                'lon_max': 48
            }
        
        print(f"Loading {mode} dataset for years {years} (Lead: {lead_weeks} weeks)...")
        
        # Preprocess data
        self.samples = preprocess_for_training(data_dir, years, ethiopia_bbox, lead_weeks=lead_weeks)
        
        # Setup normalization
        if normalizer is None:
            # Training mode: fit normalizer
            self.normalizer = S2SNormalizer()
            self.normalizer.fit(
                self.samples,
                variable_names={'input': self.input_vars, 'target': self.target_vars}
            )
            
            # Save for later use
            os.makedirs('checkpoints', exist_ok=True)
            self.normalizer.save('checkpoints/normalizer.pkl')
        else:
            # Validation/Test mode: use provided normalizer
            self.normalizer = normalizer
        
        print(f"Loaded {len(self.samples)} samples")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ds, target_ds = self.samples[idx]
        
        # Apply normalization and extract as tensors
        input_tensor = self.normalizer.transform_input(input_ds, self.input_vars)
        target_tensor = self.normalizer.transform_target(target_ds, self.target_vars)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_tensor).float()
        target_tensor = torch.from_numpy(target_tensor).float()
        
        return input_tensor, target_tensor

if __name__ == "__main__":
    ds = S2SDataset(data_dir="data/raw", years=[2000])
    print(f"Dataset Length: {len(ds)}")
    
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample shapes: X={x.shape}, Y={y.shape}")
        print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
        print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
