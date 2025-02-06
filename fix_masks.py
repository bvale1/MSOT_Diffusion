import os
import h5py
import numpy as np

dataset_path = '20250130_ImageNet_MSOT_Dataset'
with h5py.File(os.path.join(dataset_path, 'dataset.h5'), 'r+') as f:
    for split in list(f.keys()):
        print(f'Fixing masks in {split}')
        for i, sample in enumerate(list(f[split].keys())):
            print(f'{i+1}/{len(f[split].keys())}', end='\r')
            mask = f[split][sample]['bg_mask'][()]
            mask = np.flipud(mask)
            f[split][sample]['bg_mask'][()] = mask