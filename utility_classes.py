import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import h5py
import json
import os
import glob
import wandb
from torch.utils.data import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abc import abstractmethod
from typing import Union
import torch.nn as nn


class KlDivergenceStandaredNormal(nn.Module):
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


class LogScaleNormalise(object):
    '''@misc{saxena2023zeroshot,
      title={Zero-Shot Metric Depth with a Field-of-View Conditioned Diffusion Model},
      author={Saurabh Saxena and Junhwa Hur and Charles Herrmann and Deqing Sun and David J. Fleet},
      year={2023},
      eprint={2312.13252},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    '''
    def __init__(self, max_ : Union[torch.Tensor, np.ndarray, float, int], 
                 min_ : Union[torch.Tensor, np.ndarray, float, int]) -> None:
        # normalise to log scale, min and max denote the range of the overall dataset
        if isinstance(max_, float) or isinstance(max_, int):
            max_ = torch.Tensor([max_])
        elif isinstance(max_, np.ndarray):
            max_ = torch.from_numpy(max_)
        if isinstance(min_, float) or isinstance(min_, int):
            min_ = torch.Tensor([min_])
        elif isinstance(min_, np.ndarray):
            min_ = torch.from_numpy(min_)
        # max and min may be of shape (C, 1, 1), so each channel may
        # be normalised separately
        self.max_ = max_.view(-1, 1, 1)
        self.min_ = min_.view(-1, 1, 1)
        self.denom = torch.log(self.max_ / self.min_)
    
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / self.min_) / self.denom
    
    def inverse(self, tensor : torch.Tensor) -> torch.Tensor:
        return torch.exp(tensor * self.denom) * self.min_


class InstanceZeroToOneNormalise(object):
    # normalise individual sample min=0, max=1
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        min_ = torch.min(tensor).item()
        max_ = torch.max(tensor).item()
        return (tensor - min_) / (max_ - min_)
        #return 2*((tensor - min_) / (max_ - min_)) - 1
    
    def inverse(self, tensor : torch.Tensor, **kwargs) -> torch.Tensor:
        # only invert if the original min and max values are provided
        if 'min_' in kwargs and 'max_' in kwargs:
            return tensor * (kwargs['max_'] - kwargs['min_']) + kwargs['min_']
            #return ((tensor + 1) * (kwargs['min_'] - kwargs['max_']) / 2) + kwargs['min_']
        else:
            return tensor


class InstanceMeanStdNormalise(object):
    # standardise individual sample mean=0, std=1 (standard normal distribution)
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        return (tensor - mean) / std
    
    def inverse(self, tensor : torch.Tensor, **kwargs) -> torch.Tensor:
        # only invert if the original mean and std values are provided
        if 'mean' in kwargs and 'std' in kwargs:
            return (tensor * kwargs['std']) + kwargs['mean']
        else:
            return tensor
    
    
class DatasetMeanStdNormalise(object):
    # standardise to dataset mean=0, std=1 (standard normal distribution)
    def __init__(self, mean : Union[torch.Tensor, np.ndarray, float, int], 
                 std : Union[torch.Tensor, np.ndarray, float, int]) -> None:
        if isinstance(mean, float) or isinstance(mean, int):
            mean = torch.Tensor([mean])
        elif isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        if isinstance(std, float) or isinstance(std, int):
            std = torch.Tensor([std])
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        # mean and std may be of shape (C, 1, 1), so each channel may
        # be normalised separately
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
        
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std
    
    def inverse(self, tensor : torch.Tensor) -> torch.Tensor:
        # use to convert back to original scale
        return (tensor * self.std) + self.mean


class DatasetMaxMinNormalise(object):
    # normalise to entire dataset to min=0, max=1
    def __init__(self, max_ : Union[torch.Tensor, np.ndarray, float, int],
                 min_ : Union[torch.Tensor, np.ndarray, float, int]) -> None:
        if isinstance(max_, float) or isinstance(max_, int):
            max_ = torch.Tensor([max_])
        elif isinstance(max_, np.ndarray):
            max_ = torch.from_numpy(max_)
        if isinstance(min_, float) or isinstance(min_, int):
            min_ = torch.Tensor([min_])
        elif isinstance(min_, np.ndarray):
            min_ = torch.from_numpy(min_)
        # max and min may be of shape (C, 1, 1), so each channel may
        # be normalised separately
        self.max_ = max_.view(-1, 1, 1)
        self.min_ = min_.view(-1, 1, 1)
        
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        return (tensor - self.min_) / (self.max_ - self.min_)
    
    def inverse(self, tensor : torch.Tensor) -> torch.Tensor:
        # use to convert back to original scale
        return tensor * (self.max_ - self.min_) + self.min_
    

class ReconstructAbsorbtionDataset(Dataset):
    def __init__(self, data_path : str) -> None:
        super(ReconstructAbsorbtionDataset, self).__init__()
        self.path = data_path
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.cfg = json.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx : int) -> tuple:
        pass

    def plot_comparison(self, X : torch.Tensor,
                        Y : torch.Tensor,
                        Y_hat : torch.Tensor,
                        X_hat : torch.Tensor=None, # for autoencoders
                        X_transform=None,
                        Y_transform=None,
                        X_cbar_unit : str=None,
                        Y_cbar_unit : str=None,
                        **kwargs) -> tuple:
        # original sample X and reconstructed sample Y_hat
        X = X.detach().to('cpu')
        Y = Y.detach().to('cpu')
        Y_hat = Y_hat.detach().to('cpu')
        X_hat = X_hat.detach().to('cpu') if type(X_hat)==torch.Tensor else None
        if X_transform:
            if 'min_X' in kwargs and 'max_X' in kwargs:
                X = X_transform.inverse(X, min_=kwargs['min_X'], max_=kwargs['max_X'])
                X_hat = X_transform.inverse(X_hat, min_=kwargs['min_X'], max_=kwargs['max_X']) if type(X_hat)==torch.Tensor else None
            else:
                X = X_transform.inverse(X)
                X_hat = X_transform.inverse(X_hat) if type(X_hat)==torch.Tensor else None
        if Y_transform:
            if 'min_Y' in kwargs and 'max_Y' in kwargs:
                Y = Y_transform.inverse(Y, min_=kwargs['min_Y'], max_=kwargs['max_Y'])
                Y_hat = Y_transform.inverse(Y_hat, min_=kwargs['min_Y'], max_=kwargs['max_Y'])
            else:
                Y = Y_transform.inverse(Y)
                Y_hat = Y_transform.inverse(Y_hat)
        X = X.squeeze().numpy()
        Y = Y.squeeze().numpy()
        Y_hat = Y_hat.squeeze().numpy()
        X_hat = X_hat.squeeze().numpy() if type(X_hat)==torch.Tensor else None
        v_max_X = max(np.max(X), np.max(X_hat)) if type(X_hat)==np.ndarray else np.max(X)
        v_min_X = min(np.min(X), np.min(X_hat)) if type(X_hat)==np.ndarray else np.min(X)
        v_min_Y = min(np.min(Y), np.min(Y_hat))
        v_max_Y = max(np.max(Y), np.max(Y_hat))
        dx = self.cfg['dx'] * 1e3 # [m] -> [mm]
        extent = [-dx*X.shape[-2]/2, dx*X.shape[-2]/2,
                  -dx*X.shape[-1]/2, dx*X.shape[-1]/2]
        
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        img = []        
        
        img.append(axes[0, 0].imshow(
            X, cmap='binary_r', vmin=v_min_X, vmax=v_max_X,
            origin='lower', extent=extent
        ))
        axes[0, 0].set_title('X')
        
        img.append(axes[0, 1].imshow(
            Y, cmap='binary_r', vmin=v_min_Y, vmax=v_max_Y, 
            origin='lower', extent=extent
        ))
        axes[0, 1].set_title('Y')
        
        img.append(axes[0, 2].imshow(
            Y_hat, cmap='binary_r', vmin=v_min_Y, vmax=v_max_Y, 
            origin='lower', extent=extent
        ))
        axes[0, 2].set_title(r'$\hat{Y}$')
        
        residual = Y_hat - Y
        img.append(axes[1, 0].imshow(
            residual, cmap='RdBu', vmin=-np.max(np.abs(residual)),
            vmax=np.max(np.abs(residual)), origin='lower', extent=extent
        ))
        axes[1, 0].set_title(r'$\hat{Y} - Y$')
        
        cbars = []
        for i, ax in enumerate(axes.flat[:4]):
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
            cbars.append(fig.colorbar(img[i], cax=cbar_ax, orientation='vertical'))
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('z (mm)')
        if X_cbar_unit:
            cbars[0].set_label(X_cbar_unit)
        if Y_cbar_unit:
            cbars[1].set_label(Y_cbar_unit)
            cbars[2].set_label(Y_cbar_unit)
            cbars[3].set_label(Y_cbar_unit)
                
        Y_line_profile = Y[Y.shape[0]//2, :]
        Y_hat_line_profile = Y_hat[Y_hat.shape[0]//2, :]
        line_profile_axis = np.arange(-dx*Y.shape[-1]/2, dx*Y.shape[-1]/2, dx)
        axes[1, 1].plot(
            line_profile_axis, Y_line_profile, label='Y', 
            color='tab:blue', linestyle='solid'
        )
        axes[1, 1].plot(
            line_profile_axis, Y_hat_line_profile, label=r'$\hat{Y}$', 
            color='tab:red', linestyle='dashed'
        )
        axes[1, 1].set_title('Line profile')
        axes[1, 1].set_xlabel('x (mm)')
        axes[1, 1].set_ylabel(Y_cbar_unit)
        axes[1, 1].grid(True)
        axes[1, 1].set_axisbelow(True)
        axes[1, 1].set_xlim(extent[0], extent[1])
        axes[1, 1].legend()
        
        if type(X_hat) == np.ndarray:
            img.append(axes[1, 2].imshow(
                X_hat, cmap='binary_r', vmin=v_min_X, vmax=v_max_X,
                origin='lower', extent=extent
            ))
            axes[1, 2].set_title(r'$\hat{X}$')
            divider = make_axes_locatable(axes[1, 2])
            cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
            cbars.append(fig.colorbar(img[-1], cax=cbar_ax, orientation='vertical'))
            axes[1, 2].set_xlabel('x (mm)')
            axes[1, 2].set_ylabel('z (mm)')
            if X_cbar_unit:
                cbars[-1].set_label(X_cbar_unit)
        
        fig.tight_layout()
        return (fig, axes)
    
    
class SyntheticReconstructAbsorbtionDataset(ReconstructAbsorbtionDataset):
    
    def __init__(self, data_path : str,
                 split : str='train',
                 gt_type : str='mu_a',
                 data_space : str='image',
                 X_transform=None,
                 Y_transform=None) -> None:
        super(SyntheticReconstructAbsorbtionDataset, self).__init__(data_path)
        self.X_transform = X_transform
        self.Y_transform = Y_transform
        
        assert split in ['train', 'val', 'test'], f'split {split} not recognised, \
            must be "train", "val" or "test"'
        self.split = split
        
        assert gt_type in ['fluence_correction', 'mu_a'], f'gt_type {gt_type} \
            not recognised, must be "fluence_correction" or "mu_a"'
        self.gt_type = gt_type
        
        assert data_space in ['image', 'latent'], f'data_space {data_space} \
            not recognised, must be "image" or "latent"'
        self.data_space = data_space
        
        match gt_type:
            case 'fluence_correction':
                # 'corrected_image' is the image 'X' divided by the fluence 'Phi'
                self.get_Y = lambda f, sample: f[self.split][sample]['corrected_image'][()]
            case 'mu_a':
                self.get_Y = lambda f, sample: f[self.split][sample]['mu_a'][()]
        
        match data_space:
            case 'image':
                self.h5_file = os.path.join(self.path, 'dataset.h5')
            case 'latent':
                self.h5_file = os.path.join(self.path, 'embeddings.h5')
                self.get_Y = lambda f, sample: f[self.split][sample]['Y'][()]
                
        with h5py.File(self.h5_file, 'r') as f:
            self.samples = list(f[split].keys())
                
    def __getitem__(self, idx : int) -> tuple:
        with h5py.File(self.h5_file, 'r') as f:
            X = torch.from_numpy(f[self.split][self.samples[idx]]['X'][()])
            Y = torch.from_numpy(self.get_Y(f, self.samples[idx]))
        if X.dim()==2: # add channel dimension
            X = X.unsqueeze(0)
        if Y.dim()==2:
            Y = Y.unsqueeze(0)
        if self.X_transform:
            X = self.X_transform(X)
        if self.Y_transform:
            Y = self.Y_transform(Y)
        
        return (X, Y)

'''
class ReconstructAbsorbtionDatasetJanek(ReconstructAbsorbtionDataset):
    
    def __init__(self, data_path : str, 
                 split : str='train', 
                 gt_type : str='mu_a',):
        self.path = data_path # don't call super because dataset doesn't have a config file 
        
        assert split in ['train', 'val', 'test'], f'split {split} not recognised, \
            must be "train", "val" or "test"'
        self.split = split
        
        assert gt_type in ['mu_a', 'fluence_correction'], f'gt_type {gt_type} \ 
        not recognised, must be "mu_a" or "fluence_correction"'
        self.gt_type = gt_type

        match split:
            case 'train', 'val':
                npz_dirs = glob.glob(os.path.join(
                    data_path, 'training_data_1/**/*.npz'
                ), recursive=True)
                npz_dirs += glob.glob(os.path.join(
                    data_path, 'training_data_2/**/*.npz'
                ), recursive=True)
                phantoms = [npz_file.split('/')[-1] for npz_file in npz_files]
                phantoms = [phantom.split('_')[0] for phantom in phantoms]
            case 'val':
                pass
            case 'test':
                npz_files = glob.glob(os.path.join(
                    data_path, 'test_data/**/*.npz'
                ), recursive=True)

        self.samples = 
'''    

class CheckpointSaver:
    def __init__(self, dirpath : str, decreasing : bool=True, top_n : int=5,
                 wand_log : bool=False) -> None:
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value

        Code from Aman Arora's W&B report:
        https://wandb.ai/amanarora/melanoma/reports/How-to-save-all-your-trained-model-weights-locally-after-every-epoch--VmlldzoxNTkzNjY1
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.wand_log = wand_log
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model : torch.nn.Module, epoch : int, metric_val : float) -> None:
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        if self.decreasing: 
            save = metric_val<self.best_metric_val
        else:
            save = metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(
                self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing
            )
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def load_best_model(self, model : torch.nn.Module) -> None:
        if self.top_model_paths:
            try:
                model.load_state_dict(torch.load(self.top_model_paths[0]['path'], weights_only=True))
                logging.info(f"Loaded best model from {self.top_model_paths[0]['path']}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        else:
            logging.info("No models to load.")
    
    def log_artifact(self, filename : str, model_path : str, metric_val : float) -> None:
        if self.wand_log:
            artifact = wandb.Artifact(
                filename, type='model', metadata={'Validation score': metric_val}
            )
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)        
    
    def cleanup(self) -> None:
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]