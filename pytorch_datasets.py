import numpy as np
import torch, h5py, json, os, logging, glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class FluenceCorrectionDataset(Dataset):
    
    def __init__(self, path, transform=None):
        h5_dirs = glob.glob(os.path.join(path, '**/*.h5'), recursive=True)
        json_dirs = glob.glob(os.path.join(path, '**/*.json'), recursive=True)
        
        h5_dirs = {os.path.dirname(file) for file in h5_dirs}
        json_dirs = {os.path.dirname(file) for file in json_dirs}
        
        sim_dirs = h5_dirs.intersection(json_dirs)
        
        self.samples = []
        for dir in sim_dirs:
            with h5py.File(os.path.join(dir, 'data.h5'), 'r') as f:
                for key in list(f.keys()):
                    self.samples.append((dir, key))
                            
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.samples[idx][0], 'data.h5'), 'r') as f:
            X = np.array(f[self.samples[idx[1]]].get('p0_tr'))
            Y = np.array(f[self.samples[idx[1]]].get('Phi'))
            
        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)
            
        return X, Y
            