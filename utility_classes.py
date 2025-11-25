import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import h5py
import json
import os
import wandb
import glob
from torch.utils.data import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abc import abstractmethod
from typing import Union, Literal
import torch.nn as nn
from typing import Dict

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
    # standardise to dataset mean=0, std=std (standard normal distribution)
    def __init__(
            self, mean : Union[torch.Tensor, np.ndarray, float, int], 
            std : Union[torch.Tensor, np.ndarray, float, int],
            sigma_data : Union[torch.Tensor, np.ndarray, float, int]=1.0,
        ) -> None:
        if isinstance(mean, float) or isinstance(mean, int):
            mean = torch.Tensor([mean])
        elif isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        if isinstance(std, float) or isinstance(std, int):
            std = torch.Tensor([std])
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        if isinstance(sigma_data, float) or isinstance(sigma_data, int):
            sigma_data = torch.Tensor([sigma_data])
        # mean and std may be of shape (C, 1, 1), so each channel may
        # be normalised separately
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
        self.sigma_data = sigma_data.view(-1, 1, 1)

    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) * self.sigma_data / self.std
    
    def inverse(self, tensor : torch.Tensor) -> torch.Tensor:
        # use to convert back to original scale
        return (tensor * self.std / self.sigma_data) + self.mean


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
                        mask : torch.Tensor=None,
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
        Y += 1e-2 # convert mu_a from m^-1 to cm^-1
        Y_hat += 1e-2
        v_max_X = max(np.max(X), np.max(X_hat)) if type(X_hat)==np.ndarray else np.max(X)
        v_min_X = min(np.min(X), np.min(X_hat)) if type(X_hat)==np.ndarray else np.min(X)
        v_min_Y = min(np.min(Y), np.min(Y_hat))
        v_max_Y = max(np.max(Y), np.max(Y_hat))
        dx = self.cfg['dx'] * 1e3 # [m] -> [mm]
        extent = [-dx*X.shape[-2]/2, dx*X.shape[-2]/2,
                  -dx*X.shape[-1]/2, dx*X.shape[-1]/2]
        
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
        img = []        
        
        img.append(axes[0, 0].imshow(
            X, cmap='binary_r', vmin=v_min_X, vmax=v_max_X,
            origin='lower', extent=extent
        ))
        axes[0, 0].set_title(r'Reconstruction $p_{0}^{\mathrm{rec}}$')
        
        img.append(axes[0, 1].imshow(
            Y, cmap='binary_r', vmin=v_min_Y, vmax=v_max_Y, 
            origin='lower', extent=extent
        ))
        axes[0, 1].set_title(r'Reference $\mu_{\mathrm{a}}$')
        
        img.append(axes[0, 2].imshow(
            Y_hat, cmap='binary_r', vmin=v_min_Y, vmax=v_max_Y, 
            origin='lower', extent=extent
        ))
        axes[0, 2].set_title(r'Prediction $\hat{\mu_{\mathrm{a}}}$')
        
        residual = Y_hat - Y
        img.append(axes[1, 0].imshow(
            residual, cmap='RdBu', vmin=-np.max(np.abs(residual)),
            vmax=np.max(np.abs(residual)), origin='lower', extent=extent
        ))
        axes[1, 0].set_title(r'Residual $\hat{\mu_{\mathrm{a}}} - \mu_{\mathrm{a}}$')
        
        cbars = []
        for i, ax in enumerate(axes.flat[:4]):
            cbar_unit = X_cbar_unit if i==0 else Y_cbar_unit
            cbars.append(fig.colorbar(img[i], ax=ax, label=cbar_unit))
        axes[0,0].set_ylabel('z (mm)')
        axes[1,0].set_ylabel('z (mm)')
        axes[1,0].set_xlabel('x (mm)')
        axes[1,1].set_xlabel('x (mm)')
                
        Y_line_profile = Y[Y.shape[0]//2, :]
        Y_hat_line_profile = Y_hat[Y_hat.shape[0]//2, :]
        line_profile_axis = np.arange(-dx*Y.shape[-1]/2, dx*Y.shape[-1]/2, dx)
        axes[1,1].plot(
            line_profile_axis, Y_line_profile, label='Y', 
            color='tab:blue', linestyle='solid'
        )
        axes[1,1].plot(
            line_profile_axis, Y_hat_line_profile, label=r'$\hat{Y}$', 
            color='tab:red', linestyle='dashed'
        )
        axes[1,1].set_title('Line profile')
        axes[1,1].set_box_aspect(1)
        
        axes[1,1].set_ylabel(Y_cbar_unit)
        axes[1,1].grid(True)
        axes[1,1].set_axisbelow(True)
        axes[1,1].set_xlim(extent[0], extent[1])
        axes[1,1].legend()
        
        # optional plot either X_hat or mask, priority to X_hat
        if type(X_hat) == np.ndarray:
            img.append(axes[1, 2].imshow(
                X_hat, cmap='binary_r', vmin=v_min_X, vmax=v_max_X,
                origin='lower', extent=extent
            ))
            axes[1, 2].set_title(r'Reconstruction $\hat{p}_{0}^{\mathrm{rec}}$')
            cbars.append(fig.colorbar(img[-1], ax=axes[1,2], label=X_cbar_unit))
            axes[1, 2].set_xlabel('x (mm)')
                 
        elif type(mask) == torch.Tensor:
            mask = mask.detach().cpu().squeeze().numpy()
            img.append(axes[1, 2].imshow(
                mask, cmap='binary_r', origin='lower', extent=extent
            ))
            axes[1, 2].set_title('Mask')
            cbars.append(fig.colorbar(img[-1], ax=axes[1,2], label=X_cbar_unit))
            axes[1, 2].set_xlabel('x (mm)')
        
        return (fig, axes)
    
    
class SyntheticReconstructAbsorbtionDataset(ReconstructAbsorbtionDataset):
    # works for both image and latent space data, as well as synthetic 
    # images of digimouse and ImageNet digital phantoms
    def __init__(self, data_path : str,
                 split : Literal['train', 'val', 'test']='train',
                 data_space : Literal['image','latent']='image',
                 fold : Literal[0, 1, 2, 3, 4]=0,
                 X_transform=None,
                 Y_transform=None,
                 fluence_transform=None,
                 mask_transform=None) -> None:
        super(SyntheticReconstructAbsorbtionDataset, self).__init__(data_path)
        self.split = split
        self.data_space = data_space
        self.fold = fold
        self.X_transform = X_transform
        self.Y_transform = Y_transform
        self.fluence_transform = fluence_transform
        self.mask_transform = mask_transform
                
        match data_space:
            case 'image':
                self.h5_file = os.path.join(self.path, 'dataset.h5')
            case 'latent':
                self.h5_file = os.path.join(self.path, 'embeddings.h5')
                
        with h5py.File(self.h5_file, 'r') as f:
            self.samples = f[split][str(fold)]['sample_names'][:].tolist()
                
    def __getitem__(self, idx : int) -> tuple:
        with h5py.File(self.h5_file, 'r') as f:
            X = torch.from_numpy(f['samples'][self.samples[idx]]['X'][()])
            Y = torch.from_numpy(f['samples'][self.samples[idx]]['mu_a'][()])
            fluence = torch.from_numpy(f['samples'][self.samples[idx]]['Phi'][()])
            bg_mask = torch.from_numpy(f['samples'][self.samples[idx]]['bg_mask'][()])
            wavelength_nm = f['samples'][self.samples[idx]]['wavelength_nm'][()]
        wavelength_nm = torch.tensor([wavelength_nm], dtype=torch.int)    
        
        if X.dim()==2: # add channel dimension
            X = X.unsqueeze(0)
        if Y.dim()==2:
            Y = Y.unsqueeze(0)
        if fluence.dim()==2:
            fluence = fluence.unsqueeze(0)
        if bg_mask.dim()==2:
            bg_mask = bg_mask.unsqueeze(0)
            
        if self.X_transform:
            X = self.X_transform(X)
        if self.Y_transform:
            Y = self.Y_transform(Y)
        if self.fluence_transform:
            fluence = self.fluence_transform(fluence)
        if self.mask_transform:
            bg_mask = self.mask_transform(bg_mask)
        
        return (X, Y, fluence, wavelength_nm, bg_mask, torch.zeros_like(bg_mask), self.samples[idx])
    

class e2eQPATReconstructAbsorbtionDataset(ReconstructAbsorbtionDataset):
    # for end-to-end QPAT experimental dataset
    
    # Randomised but reproducible fold-partitions for the 84 phantoms in the training data set
    # (same as in the paper)
    folds = {
        0: [ 2, 79,  5, 66, 55, 45, 62, 26, 18, 75, 73, 24, 39, 36, 48, 33],
        1: [37, 67, 13, 71,  3,  1, 69, 78, 54, 72, 11, 25, 34, 40, 12, 51],
        2: [19, 30, 83, 57, 74, 53, 41, 82, 20, 31, 28, 76, 81, 64, 42, 52],
        3: [65, 43,  6, 68, 15,  8,  4, 17, 44, 14, 27, 23, 80, 56,  0, 49],
        4: [38, 63, 32, 60, 29, 35,  9, 21, 22, 47, 10, 77, 61, 50,  7, 59]
    }
    '''
    https://github.com/BohndiekLab/end_to_end_phantom_QPAT
    @article{Janek2023IEEE,
    author = {Janek Gröhl and Thomas R Else and Lina Hacker and Ellie V Bunce and Paul W Sweeney and Sarah E Bohndiek},
    journal = {IEEE Transactions on Medical Imaging},
    publisher = {IEEE},
    title = {Moving beyond simulation: data-driven quantitative photoacoustic imaging using tissue-mimicking phantoms},
    year = {2023},
    }
    @article{grohl2023dataset,
    title={Dataset for: Moving beyond simulation: data-driven quantitative photoacoustic imaging using tissue-mimicking phantoms},
    author={Gr{\"o}hl, Janek and Else, Thomas and Hacker, Lina and Bunce, Ellie and Sweeney, Paul and Bohndiek, Sarah},
    year={2023}
    }
    '''     
    def __init__(self, data_path : str,
                 stats : dict,
                 fold : Literal[0, 1, 2, 3, 4],
                 train : bool,
                 augment : bool,
                 use_all_data : bool,
                 experimental_data : bool=True, 
                 shuffle : bool=False,
                 X_transform : callable=None,
                 Y_transform : callable=None,
                 fluence_transform : callable=None,
                 mask_transform : callable=None) -> None:
        
        vars(self).update(locals())
        self.cfg = stats
        
        files = glob.glob(data_path + "/*.npz")
        files.sort()
        if not use_all_data:
            tmp_files = []
            if train:
                for idx in range(int(len(files)/21)):
                    if not idx in self.folds[fold]:
                        tmp_files += files[idx*21:(idx+1)*21]
            else:
                for idx in range(int(len(files) / 21)):
                    if idx in self.folds[fold]:
                        tmp_files += files[idx * 21:(idx + 1) * 21]
            files = tmp_files
        self.files = files
        # without shuffling each batch will mostly contain images of the same
        # sample but at different wavelengths, shuffling may reduce overfitting
        if shuffle:
            rng = np.random.RandomState(42)
            rng.shuffle(self.files)
        print(f"Found {len(files)} items.")
        
    def __len__(self):
        if self.train and self.augment:
            return len(self.files) * 2
        else:
            return len(self.files)
        
    def __getitem__(self, idx : int) -> tuple:
        # every other sample is the same as the previous one but flipped
        np_data = np.load(self.files[idx // 2])
        if self.experimental_data:
            signal = torch.from_numpy(np_data["features_das"].reshape(1, 288, 288)).float()
        else:
            signal = torch.from_numpy(np_data["features_sim"].reshape(1, 288, 288)).float()
            
        segmentation = np_data["segmentation"]
        # 0 == coupling medium, 1 == sample_background, 1 < inclusions
        if self.stats['segmentation']['plus_one']:
            segmentation = segmentation + 1       
        bg_mask = torch.from_numpy(segmentation).int().unsqueeze(0) == 1
        inclusion_mask = torch.from_numpy(segmentation).int().unsqueeze(0) > 1
        absorption = torch.from_numpy(np_data["mua"].reshape(1, 288, 288)).float()
        fluence = torch.from_numpy(np_data["fluence"].reshape(1, 288, 288)).float()
        wavelength_nm = int(self.files[idx // 2].split('_')[-1][:3])
        wavelength_nm = torch.tensor([wavelength_nm], dtype=torch.int)
            
        if self.X_transform:
            signal = self.X_transform(signal)
        if self.Y_transform:
            absorption = self.Y_transform(absorption)
        if self.fluence_transform:
            fluence = self.fluence_transform(fluence)
        if self.mask_transform:
            bg_mask = self.mask_transform(bg_mask)
            inclusion_mask = self.mask_transform(inclusion_mask)
        
        # every other sample is the same as the previous one but flipped
        if self.train and self.augment and (idx % 2 == 1):
            signal = torch.fliplr(signal)
            absorption = torch.fliplr(absorption)
            fluence = torch.fliplr(fluence)
            bg_mask = torch.fliplr(bg_mask)
            inclusion_mask = torch.fliplr(inclusion_mask)
        
        return (signal, absorption, fluence, wavelength_nm, bg_mask, inclusion_mask, self.files[idx // 2])


class CombineMultipleDatasets(Dataset):
    def __init__(self, datasets : Dict[str, Dataset], seed : int=42) -> None:
        """use to train on multiple datasets at once, samples from each dataset 
        are shuffled and concatenated together.

        Args:
            datasets (List[Dataset]): list of pytorch datasets to combine
            seed (int, optional): seed for shuffling the dataset. Defaults to 42.
        """
        super(CombineMultipleDatasets, self).__init__()
        self.datasets = datasets
        self.seed = seed
        # Each sample in the combined dataset is a tuple of (dataset_name, sample_idx)
        self.samples = []
        for dataset_name, dataset in datasets.items():
            dataset_samples = [(dataset_name, i) for i in range(len(dataset))]
            self.samples.extend(dataset_samples)
        # Shuffle the combined dataset
        rng = np.random.RandomState(seed)
        rng.shuffle(self.samples)
        
    def __len__(self) -> int:
        return sum([d.__len__() for d in list(self.datasets.values())])
    
    def __getitem__(self, idx : int) -> tuple:
        dataset_name, sub_idx = self.samples[idx]
        sample = self.datasets[dataset_name].__getitem__(sub_idx)
        return sample
    
    
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
            logging.info(f"Current metric value {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
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
        

class TestMetricCalculator():
    # class to evaluate test metrics over the entire test set, which is passed
    # through in batches
    def __init__(self) -> None:
        self.metrics = {
            'RMSE' : [],
            'MAE' : [],
            'Rel_Err' : [],
            'PSNR' : [],
            'SSIM' : [],
            'R2' : []
        }        
    
    def __call__(self, Y : torch.Tensor, Y_hat : torch.Tensor,
                 Y_transform=None, Y_mask=None) -> None:
        assert Y.shape == Y_hat.shape, f"Y.shape {Y.shape} must equal \
            Y_hat.shape {Y_hat.shape}"
        assert Y.dim() == 4, f"Y.dim() {Y.dim()} must be of shape (b, c, h, w)"
        b = Y.shape[0]
        Y = Y.detach().cpu()
        Y_hat = Y_hat.detach().cpu()
        if Y_transform:
            Y = Y_transform.inverse(Y)
            Y_hat = Y_transform.inverse(Y_hat)
        Y = Y.view(b, -1) # [b, c*h*w]
        Y_hat = Y_hat.view(b, -1) # [b, c*h*w]
        if type(Y_mask) == torch.Tensor:
            Y_mask = Y_mask.detach().cpu().view(b, -1) # [b, c*h*w]
            Y_mask_sum = Y_mask.sum(dim=1, keepdim=True) # [b, 1]
            Y_max = (Y*Y_mask).amax(dim=1, keepdim=True) # [b, 1]
        else:
            Y_max = Y.amax(dim=1, keepdim=True)
        
        if type(Y_mask) == torch.Tensor:
            # [b, c*h*w] * [b, c*h*w] = [b, c*h*w] -> [b, 1]
            RMSE = torch.sqrt((((Y - Y_hat)*Y_mask)**2).sum(dim=1, keepdim=True) / Y_mask_sum)
            MAE = torch.abs((Y - Y_hat)*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            Rel_Err = 100 * torch.abs((Y - Y_hat)*Y_mask/Y).sum(dim=1, keepdim=True) / Y_mask_sum
            mean_Y = (Y*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            mean_Y_hat = (Y_hat*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            var_Y = (((Y - mean_Y)**2)*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            var_Y_hat = (((Y_hat - mean_Y_hat)**2)*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            cov_Y_Y_hat = ((Y - mean_Y)*(Y_hat - mean_Y_hat)*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            SSr = (((Y - Y_hat)**2)*Y_mask).sum(dim=1, keepdim=True) # sum of squares of residuals
            SSt = (((Y - mean_Y)**2)*Y_mask).sum(dim=1, keepdim=True) # total sum of squares
        else:
            # [b, c*h*w] * [b, c*h*w] = [b, c*h*w] -> [b, 1]
            RMSE = torch.sqrt(torch.mean((Y - Y_hat)**2, dim=1, keepdim=True))
            MAE = torch.mean(torch.abs(Y - Y_hat), dim=1, keepdim=True)
            Rel_Err = torch.mean(100 * torch.abs(Y - Y_hat) / Y, dim=1, keepdim=True)
            mean_Y = torch.mean(Y, dim=1, keepdim=True)
            mean_Y_hat = torch.mean(Y_hat, dim=1, keepdim=True)
            var_Y = torch.var(Y, dim=1, keepdim=True)
            var_Y_hat = torch.var(Y_hat, dim=1, keepdim=True)
            cov_Y_Y_hat = torch.mean(
                (Y - mean_Y)*(Y_hat - mean_Y_hat), dim=1, keepdim=True
            )
            SSr = torch.sum((Y - Y_hat)**2, dim=1, keepdim=True) # sum of squares of residuals
            SSt = torch.sum((Y - mean_Y)**2, dim=1, keepdim=True) # total sum of squares
            
        PSNR = 20*torch.log10(Y_max / RMSE)
        c1 = (0.01 * Y_max)**2
        c2 = (0.03 * Y_max)**2
        SSIM = (2*mean_Y*mean_Y_hat + c1)*(2*cov_Y_Y_hat + c2) / \
            ((mean_Y**2 + mean_Y_hat**2 + c1)*(var_Y + var_Y_hat + c2))
        R2 = 1 - (SSr / SSt)
        
        self.metrics['RMSE'] += RMSE.squeeze().tolist()
        self.metrics['MAE'] += MAE.squeeze().tolist()
        self.metrics['Rel_Err'] += Rel_Err.squeeze().tolist()
        self.metrics['PSNR'] += PSNR.squeeze().tolist()
        self.metrics['SSIM'] += SSIM.squeeze().tolist()
        self.metrics['R2'] += R2.squeeze().tolist()
                
    def get_metrics(self) -> dict:
        return {
            'mean_RMSE' : np.nanmean(np.asarray(self.metrics['RMSE'])),
            'std_RMSE' : np.nanstd(np.asarray(self.metrics['RMSE'])),
            'mean_MAE' : np.nanmean(np.asarray(self.metrics['MAE'])),
            'std_MAE' : np.nanstd(np.asarray(self.metrics['MAE'])),
            'mean_Rel_Err' : np.nanmean(np.asarray(self.metrics['Rel_Err'])),
            'std_Rel_Err' : np.nanstd(np.asarray(self.metrics['Rel_Err'])),
            'mean_PSNR' : np.nanmean(np.asarray(self.metrics['PSNR'])),
            'std_PSNR' : np.nanstd(np.asarray(self.metrics['PSNR'])),
            'mean_SSIM' : np.nanmean(np.asarray(self.metrics['SSIM'])),
            'std_SSIM' : np.nanstd(np.asarray(self.metrics['SSIM'])),
            'mean_R2' : np.nanmean(np.asarray(self.metrics['R2'])),
            'std_R2' : np.nanstd(np.asarray(self.metrics['R2']))
        }
        
    def save_metrics_all_test_samples(self, save_path : str) -> None:
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)


class LoRaFineTuneModule(nn.Module):
    def __init__(self, 
                 module : nn.Module,
                 r : int=4,
                 alpha : float=1.0,
                 leaky_relu_slope : float=0.0,
                 verbose : bool=True) -> None:
        super(LoRaFineTuneModule, self).__init__()
        self.module = module
        self.r = r
        self.alpha = alpha
        self.leaky_relu_slope = leaky_relu_slope
        self.verbose = verbose
        self.lora_names = {}

        for name, param in self.module.named_parameters():
            if verbose:
                logging.info(f'param name: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}')
            if param.ndim == 2: # linear layer
                d, n = param.shape
                A = torch.zeros((r, n), device=param.device)
                B = torch.zeros((d, r), device=param.device)
            elif param.ndim == 4: # conv2d layer
                # (out_channels, in_channels, kernel_h, kernel_w)
                cout, cin, kh, kw = param.shape
                A = torch.zeros((r, cin, kh), device=param.device)
                B = torch.zeros((cout, kw, r), device=param.device)
            else:
                continue  # skip non-linear/conv2d layers
            # add LoRa parameters A and B to module
            # initialize B with kaiming normal, A with zeros
            nn.init.zeros_(B)
            nn.init.kaiming_uniform_(A, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
            # For conv2d: fan_in = cin * kh (receptive field per output element)
            # For linear: fan_in = n (number of input features)
            
            if self.verbose:
                logging.info(f'Adding LoRa parameters for {name}: A shape {A.shape}, B shape {B.shape}')
            name_no_dot = name.replace('.', '_') # cannot have '.' in parameter names
            self.lora_names[name] = (f'lora_{name_no_dot}_B', f'lora_{name_no_dot}_A')
            self.register_parameter(f'lora_{name_no_dot}_A', nn.Parameter(A))
            self.register_parameter(f'lora_{name_no_dot}_B', nn.Parameter(B))
            
    def eval(self):
        self.module.eval()
        return self
    
    def train(self, mode: bool = True):
        self.module.train(mode)
        for name, (name_B, name_A) in self.lora_names.items():
            if self.verbose:
                logging.info(f'Setting requires_grad for {name}, {name_B}, {name_A}')
            self.module.get_parameter(name).requires_grad = False
            self.get_parameter(name_B).requires_grad = True
            self.get_parameter(name_A).requires_grad = True
        return self

    def forward(self, *args, **kwargs):
        for name, (name_B, name_A) in self.lora_names.items():
            
            if self.module.get_parameter(name).ndim==4:
                # (cout, kw, r)(r, cin, kh) -> (cout, cin, kernel_h, kernel_w)
                self.module.get_parameter(name).data += self.alpha * torch.einsum('ijr,rkl->ikjl', self.get_parameter(name_B).data, self.get_parameter(name_A).data)
            else:
                # (d, r)(r, n) -> (d, n)
                self.module.get_parameter(name).data += self.alpha * (self.get_parameter(name_B).data @ self.get_parameter(name_A).data)

        output = self.module.forward(*args, **kwargs)

        for name, (name_B, name_A) in self.lora_names.items():
            if self.module.get_parameter(name).ndim==4:
                # remove LoRa update
                self.module.get_parameter(name).data -= self.alpha * torch.einsum('ijr,rkl->ikjl', self.get_parameter(name_B).data, self.get_parameter(name_A).data)
            else:
                self.module.get_parameter(name).data -= self.alpha * (self.get_parameter(name_B).data @ self.get_parameter(name_A).data)

        return output


class OrthogonalFineTuneModule():
    def __init__(self, 
                 module : nn.Module,
                 r : int=4,
                 verbose : bool=True) -> nn.Module:
        self.module = module
        self.oft_dict = {}
        self.state_dict = self.module.state_dict()
        for name, tensor in self.state_dict.items():
            if verbose:
                logging.info(f'param name: {name}, shape: {tensor.shape}, requires_grad: {tensor.requires_grad}')
            if tensor.ndim == 2: # linear layer
                n_blocks = r
            if tensor.ndim == 4: # conv2d layer
                # n_blocks = number of convolutional neurons in layer
                n_blocks = tensor.shape[1]
            # add orthogonal fine-tuning paramerter
            self.oft_dict[name] = []
            for block in range(r):
                # add parameter matrix R of shape (out_features/r, in_features/r) to module
                R = torch.zeros((tensor.shape[0]//r, tensor.shape[0]//r), device=tensor.device, requires_grad=True)
                module.register_parameter(f'oft_{name}_block{block}', nn.Parameter(R))
                self.oft_dict[name].append(R)

        return self.module

    def forward(self, *args, **kwargs):
        for name in self.state_dict.keys():
            self.state_dict[name].require_grad = False
            R = torch.block_diag(*self.oft_dict[name])
            self.state_dict[name] = R @ self.state_dict[name]

        return self.module.forward(*args, **kwargs)

        
    