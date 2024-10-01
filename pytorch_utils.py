import numpy as np
import torch, h5py, json, os
from torch.utils.data import Dataset
from abc import abstractmethod


class ReplaceNaNWithZero(object):
    def __call__(self, tensor : torch.Tensor):
        tensor[torch.isnan(tensor)] = 0.0
        return tensor
    
    
class MeanStdNormalise(object):
    
    def __init__(self, mean : torch.Tensor, std : torch.Tensor):
        self.mean = mean.unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(-1).unsqueeze(-1)
        
    def __call__(self, tensor : torch.Tensor):
        return (tensor - self.mean) / self.std
    
    def inverse(self, tensor : torch.Tensor):
        # use to convert back to original scale
        return (tensor * self.std) + self.mean


class ReconstructAbsorbtionDataset(Dataset):
    
    def __init__(self, path, gt_type='mu_a', X_transform=None, Y_transform=None):
        self.path = path
        self.X_transform = X_transform
        self.Y_transform = Y_transform
        
        with os.path.join(path, 'config.json') as f:
            self.cfg = json.load(f)
        
        if gt_type == 'fluence_correction':
            self.gt_type = 'fluence_correction'
            self.get_Y = lambda f, sample: f[sample]['Y'][()]
        elif gt_type == 'mu_a':
            self.gt_type = 'mu_a'
            self.get_Y = lambda f, sample: f[sample]['mu_a'][()]
        else:
            raise ValueError(
                f'gt_type {gt_type} not recognised, must be "fluence_correction" or "mu_a"'
            )
        
        with h5py.File(os.path.join(self.path, 'data.h5'), 'r') as f:
            self.samples = list(f.keys())
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.path, 'data.h5'), 'r') as f:
            X = torch.from_numpy(f[self.samples[idx]]['X'][()])
            Y = torch.from_numpy(self.get_Y(f, self.samples[idx]))
            
        if self.X_transform:
            X = self.X_transform(X)
        
        if self.Y_transform:
            Y = self.Y_transform(Y)
            
        return X, Y
    
    @abstractmethod
    def plot_sample(self, idx, pred):
        pass