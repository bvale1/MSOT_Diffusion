import numpy as np
import matplotlib.pyplot as plt
import torch, h5py, json, os
from torch.utils.data import Dataset
from abc import abstractmethod
from mpl_toolkits.axes_grid1 import make_axes_locatable


class KlDivergenceStandaredNormal(torch.nn.Module):
    '''KL divergence between a normal distribution and a standard normal distribution.
    Also uses batch mean accumulation'''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, mu : torch.Tensor, log_var : torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


class ReplaceNaNWithZero(object):
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        tensor[torch.isnan(tensor)] = 0.0
        return tensor
    
    
class MeanStdNormalise(object):
    def __init__(self, mean : torch.Tensor, std : torch.Tensor) -> None:
        self.mean = mean.unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(-1).unsqueeze(-1)
        
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std
    
    def inverse(self, tensor : torch.Tensor) -> torch.Tensor:
        # use to convert back to original scale
        return (tensor * self.std) + self.mean


class ReconstructAbsorbtionDataset(Dataset):
    
    def __init__(self, data_path : str,
                 gt_type : str='mu_a',
                 X_transform=None,
                 Y_transform=None) -> None:
        self.path = data_path
        self.X_transform = X_transform
        self.Y_transform = Y_transform
        
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
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
        
        with h5py.File(os.path.join(self.path, 'dataset.h5'), 'r') as f:
            self.samples = list(f.keys())
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx : int) -> tuple:
        with h5py.File(os.path.join(self.path, 'dataset.h5'), 'r') as f:
            X = torch.from_numpy(f[self.samples[idx]]['X'][()])
            Y = torch.from_numpy(self.get_Y(f, self.samples[idx]))
        if self.X_transform:
            X = self.X_transform(X)
        if self.Y_transform:
            Y = self.Y_transform(Y)
        return (X, Y)
    
    def plot_comparison(self, X : torch.Tensor,
                        X_hat : torch.Tensor, 
                        transform=None,
                        cbar_unit : str=None) -> tuple:
        # original sample X and reconstructed sample X_hat
        if transform:
            X = transform.inverse(X)
            X_hat = transform.inverse(X_hat)
        X = X.squeeze().detach().numpy()
        X_hat = X_hat.squeeze().detach().numpy()
        v_min = min(np.min(X), np.min(X_hat))
        v_max = max(np.max(X), np.max(X_hat))
        dx = self.cfg['dx'] * 1e3 # [m] -> [mm]
        extent = [-dx*X.shape[-2]/2, dx*X.shape[-2]/2,
                  -dx*X.shape[-1]/2, dx*X.shape[-1]/2]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        img = []        
        
        img.append(axes[0, 0].imshow(
            X, cmap='binary_r', vmin=v_min, vmax=v_max,
            origin='lower', extent=extent
        ))
        axes[0, 0].set_title('X')
        
        img.append(axes[0, 1].imshow(
            X_hat, cmap='binary_r', vmin=v_min, vmax=v_max, 
            origin='lower', extent=extent
        ))
        axes[0, 1].set_title(r'$\hat{X}$')
        
        residual = X - X_hat
        img.append(axes[1, 0].imshow(
            residual, cmap='RdBu', vmin=-np.max(np.abs(residual)),
            vmax=np.max(np.abs(residual)), origin='lower', extent=extent
        ))
        axes[1, 0].set_title(r'$X - \hat{X}$')
        
        for i, ax in enumerate(axes.flat[:3]):
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(img[i], cax=cbar_ax, orientation='vertical')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('z (mm)')
            if cbar_unit:
                cbar.set_label(cbar_unit)
                
        X_line_profile = X[X.shape[0]//2, :]
        X_hat_line_profile = X_hat[X_hat.shape[0]//2, :]
        line_profile_axis = np.arange(-dx*X.shape[-1]/2, dx*X.shape[-1]/2, dx)
        axes[1, 1].plot(
            line_profile_axis, X_line_profile, label='X', 
            color='tab:blue', linestyle='solid'
        )
        axes[1, 1].plot(
            line_profile_axis, X_hat_line_profile, label=r'$\hat{X}$', 
            color='tab:red', linestyle='dashed'
        )
        axes[1, 1].set_title('Line profile')
        axes[1, 1].set_xlabel('x (mm)')
        axes[1, 1].set_ylabel(cbar_unit)
        axes[1, 1].grid(True)
        axes[1, 1].set_axisbelow(True)
        axes[1, 1].set_xlim(extent[0], extent[1])
        axes[1, 1].legend()
        fig.tight_layout()
        return (fig, axes)
