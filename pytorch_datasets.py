import numpy as np
import torch, h5py, json, os
from torch.utils.data import Dataset


class FluenceCorrectionDataset(Dataset):
    
    def __init__(self, path, gt_type='fluence_correction', X_transform=None, Y_transform=None):
        self.path = path
        self.X_transform = X_transform
        self.Y_transform = Y_transform
        
        with os.path.join(path, 'config.json') as f:
            self.cfg = json.load(f)
        
        if gt_type == 'fluence_correction':
            self.gt_type = 'fluence_correction'
            self.get_Y = lambda f, sample: torch.from_numpy(f[sample]['Y'][()])
        elif gt_type == 'mu_a':
            self.gt_type = 'mu_a'
            self.get_Y = lambda f, sample: torch.from_numpy(f[sample]['mu_a'][()])
        
        with h5py.File(os.path.join(self.path, 'data.h5'), 'r') as f:
            self.samples = list(f.keys())
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.path, 'data.h5'), 'r') as f:
            X = torch.from_numpy(f[self.samples[idx]]['X'][()])
            Y = self.get_Y(f, self.samples[idx])
            
        if self.X_transform:
            X = self.X_transform(X)
        
        if self.Y_transform:
            Y = self.Y_transform(Y)
            
        return X, Y
            