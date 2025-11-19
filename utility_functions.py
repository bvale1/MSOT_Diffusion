import wandb
import json
import yaml
import os
import logging
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Callable, Union

from utility_classes import *


def get_config() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--synthetic_or_experimental', choices=['experimental', 'synthetic', 'both'], default='synthetic', help='whether to use synthetic or experimental data')
    parser.add_argument('--experimental_root_dir', type=str, default='/home/billy/Datasets/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms/', help='path to the root directory of the experimental dataset.')
    parser.add_argument('--synthetic_root_dir', type=str, default='/home/billy/Datasets/20250716_ImageNet_MSOT_Dataset/', help='path to the root directory of the synthetic dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs, set to zero or one to test code')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='batch size for inference, 4x train_batch_size should have similar device memory requirements')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--save_test_examples', help='save test examples to save_dir and wandb', action='store_true', default=False)
    parser.add_argument('--wandb_log', help='use wandb logging', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='Unet_checkpoints', help='path to save the model')
    parser.add_argument('--load_checkpoint_dir', type=str, default=None, help='path to load a model checkpoint')
    parser.add_argument('--warmup_period', type=int, default=1, help='warmup period for the learning rate, must be int greater than 0')
    parser.add_argument('--model', choices=['UNet_e2eQPAT', 'Swin_UNet', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM', 'DiT', 'EDM2'], default='EDM2', help='model to train')
    parser.add_argument('--data_normalisation', choices=['standard', 'minmax'], default='standard', help='normalisation method for the data')
    parser.add_argument('--fold', choices=['0', '1', '2', '3', '4'], default='0', help='fold for cross-validation, only used for experimental data')
    parser.add_argument('--wandb_notes', type=str, default='None', help='optional, comment for wandb')
    parser.add_argument('--predict_fluence', default=False, help='predict fluence as well as mu_a', action='store_true')
    parser.add_argument('--no_lr_scheduler', default=False, help='do not use lr scheduler', action='store_true')
    parser.add_argument('--freeze_encoder', default=False, help='freeze the encoder', action='store_true')
    parser.add_argument('--objective', choices=['pred_v', 'pred_noise'], default='pred_v', help='DDIM only, objective of the diffusion model')
    parser.add_argument('--self_condition', default=False, help='DDIM only, condition on the previous timestep', action='store_true')
    parser.add_argument('--attention', default=False, help='diffusion UNet only, use attention heads', action='store_true')
    parser.add_argument('--std_data', type=float, default=0.5, help='standard normalisation only, expected std of the data after normalisation')
    parser.add_argument('--phema_reconstruction_std', type=float, default=0.07, help='ema std to reconstruct for validation and testing epochs')
    parser.add_argument('--lora_rank', type=int, default=0, help='rank for LoRA layers, 0 means no LoRA')

    args = parser.parse_args()
    var_args = vars(args)
    logging.info(f'args dict: {var_args}')

    return (args, var_args)


def square_centre_crop(image : np.ndarray, size : int) -> np.ndarray:
    width, height = image.shape[-2:]
    if width < size or height < size:
        print('Image is smaller than crop size, returning original image')
        return image
    else:
        x = (width - size) // 2
        y = (height - size) // 2
        image = image[..., x:x+size, y:y+size]
        return image
    

def get_best_and_worst(loss : torch.Tensor, 
                       best_and_worst_examples : dict, 
                       arg0_idx : int) -> dict:
    if loss.min().item() < best_and_worst_examples['best']['loss']:
        best_and_worst_examples['best']['loss'] = loss.min().item()
        best_and_worst_examples['best']['index'] = arg0_idx+loss.argmin().item()
    if loss.max().item() > best_and_worst_examples['worst']['loss']:
        best_and_worst_examples['worst']['loss'] = loss.max().item()
        best_and_worst_examples['worst']['index'] = arg0_idx+loss.argmax().item()
    return best_and_worst_examples


def remove_softmax(module : nn.Module) -> None:
    '''
    Remove all Softmax layers from a model.
    '''
    for name, m in module.named_children():
        if isinstance(m, torch.nn.Softmax):
            setattr(module, name, torch.nn.Identity())
        elif hasattr(m, 'children'):
            remove_softmax(m)
            

def remove_dropout(module : nn.Module) -> None:
    '''
    Remove all Dropout layers from a model.
    '''
    for name, m in module.named_children():
        if isinstance(m, torch.nn.Dropout):
            setattr(module, name, torch.nn.Identity())
        elif hasattr(m, 'children'):
            remove_dropout(m)
            

def remove_batchnorm(module : nn.Module) -> None:
    '''
    Remove all BatchNorm and LayerNorm layers from a model.
    '''
    for name, m in module.named_children():
        if isinstance(m, torch.nn.BatchNorm2d):
            setattr(module, name, torch.nn.Identity())
        elif hasattr(m, 'children'):
            remove_batchnorm(m)
            

def replace_batchnorm_with_groupnorm(module : nn.Module, ch_per_group : int=16) -> None:
    '''
    Replace all BatchNorm layers with GroupNorm layers.
    https://amaarora.github.io/posts/2020-08-09-groupnorm.html
    '''
    for name, m in module.named_children():
        if isinstance(m, torch.nn.BatchNorm2d):
            if m.num_features % ch_per_group != 0:
                setattr(module, name, torch.nn.GroupNorm(
                    m.num_features, m.num_features
                ))
            else:
                setattr(module, name, torch.nn.GroupNorm(
                    m.num_features//ch_per_group, m.num_features
                ))
        elif hasattr(m, 'children'):
            replace_batchnorm_with_groupnorm(m, ch_per_group)

    
def reset_weights(module : nn.Module) -> None:
    '''
    Reset all the parameters of a model.
    '''
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    else:
        for m in module.children():
            reset_weights(m)
            

def define_transforms(args : argparse.Namespace, normalise_params) -> tuple:
    match args.data_normalisation:
        case 'minmax':
            normalise_x = DatasetMaxMinNormalise(
                torch.Tensor([normalise_params['normalisation_X']['max']]),
                torch.Tensor([normalise_params['normalisation_X']['min']]),
            )
            normalise_mu_a = DatasetMaxMinNormalise(
                torch.Tensor([normalise_params['normalisation_mu_a']['max']]),
                torch.Tensor([normalise_params['normalisation_mu_a']['min']]),
            )
            normalise_fluence = DatasetMaxMinNormalise(
                torch.Tensor([normalise_params['normalisation_Phi']['max']]),
                torch.Tensor([normalise_params['normalisation_Phi']['min']]),
            )
            
        case 'standard':
            normalise_x = DatasetMeanStdNormalise(
                torch.Tensor([normalise_params['normalisation_X']['mean']]),
                torch.Tensor([normalise_params['normalisation_X']['std']]),
                sigma_data=args.std_data,
            )
            normalise_mu_a = DatasetMeanStdNormalise(
                torch.Tensor([normalise_params['normalisation_mu_a']['mean']]),
                torch.Tensor([normalise_params['normalisation_mu_a']['std']]),
                sigma_data=args.std_data,
            )
            normalise_fluence = DatasetMeanStdNormalise(
                torch.Tensor([normalise_params['normalisation_Phi']['mean']]),
                torch.Tensor([normalise_params['normalisation_Phi']['std']]),
                sigma_data=args.std_data,
            )
    x_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        transforms.Resize(
            (args.image_size, args.image_size), 
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        normalise_x
    ])
    mu_a_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        normalise_mu_a
    ])
    fluence_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        normalise_fluence
    ])
    mask_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
    ])
    transforms_dict = {
        'x_transform' : x_transform,
        'normalise_x' : normalise_x,
        'mu_a_transform' : mu_a_transform,
        'normalise_mu_a' : normalise_mu_a,
        'fluence_transform' : fluence_transform,
        'normalise_fluence' : normalise_fluence,
        'mask_transform' : mask_transform
    }
    return transforms_dict
    
def create_synthetic_dataloaders(args : argparse.Namespace,
                                 model_name : str) -> tuple:
    if args.wandb_log and not wandb.run:
        wandb.login()
        wandb.init(
            project='MSOT_Diffusion', name=f'{model_name}_fold{args.fold}', 
            save_code=True, reinit=True, config=vars(args), notes=args.wandb_notes
        )
    
    with open(os.path.join(args.synthetic_root_dir, 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    transforms_dict = define_transforms(args, config)
    
    datasets = {
        'train' : SyntheticReconstructAbsorbtionDataset(
            args.synthetic_root_dir, split='train', data_space='image', fold=args.fold,
            X_transform=transforms_dict['x_transform'], 
            Y_transform=transforms_dict['mu_a_transform'], 
            fluence_transform=transforms_dict['fluence_transform'],
            mask_transform=transforms_dict['mask_transform']
        ),
        'val' : SyntheticReconstructAbsorbtionDataset(
            args.synthetic_root_dir, split='val', data_space='image', fold=args.fold,
            X_transform=transforms_dict['x_transform'], 
            Y_transform=transforms_dict['mu_a_transform'], 
            fluence_transform=transforms_dict['fluence_transform'],
            mask_transform=transforms_dict['mask_transform']
        ),
        'test' : SyntheticReconstructAbsorbtionDataset(
            args.synthetic_root_dir, split='test', data_space='image', fold=args.fold,
            X_transform=transforms_dict['x_transform'], 
            Y_transform=transforms_dict['mu_a_transform'], 
            fluence_transform=transforms_dict['fluence_transform'],
            mask_transform=transforms_dict['mask_transform']
        ),
    }
    dataloaders = {
        'train' : DataLoader(
            datasets['train'], batch_size=args.train_batch_size, shuffle=False, num_workers=4
        ),
        # backpropagation not performed on the validation set so batch size can be larger
        'val' : DataLoader( 
            datasets['val'], batch_size=args.val_batch_size, shuffle=False, num_workers=4
        ),
        # backpropagation not performed on the test set so batch size can be larger
        'test' : DataLoader(
            datasets['test'], batch_size=args.val_batch_size, shuffle=False, num_workers=4
        )
    }
    logging.info(f'train: {len(datasets['train'])}, val: {len(datasets['val'])}, \
        test: {len(datasets["test"])}')
    return (datasets, dataloaders, transforms_dict)


def create_e2eQPAT_dataloaders(args : argparse.Namespace,
                               model_name : str, 
                               stats_path : str) -> tuple:
    if args.wandb_log and not wandb.run:
        wandb.login()
        wandb.init(
            project='MSOT_EDM2', name=f'{model_name}_fold{args.fold}', 
            save_code=True, reinit=True, config=vars(args), notes=args.wandb_notes
        )
    
    with open(stats_path, 'r') as f:
        stats = json.load(f) # <- dataset config contains normalisation parameters
    
    transforms_dict = define_transforms(args, stats)
    
    datasets = {
        'train' : e2eQPATReconstructAbsorbtionDataset(
            os.path.join(args.experimental_root_dir, 'training'),
            stats=stats, fold=int(args.fold), train=True, augment=True,
            use_all_data=False, experimental_data=True, shuffle=False,
            X_transform=transforms_dict['x_transform'], 
            Y_transform=transforms_dict['mu_a_transform'], 
            fluence_transform=transforms_dict['fluence_transform'],
            mask_transform=transforms_dict['mask_transform']
        ),
        'val' : e2eQPATReconstructAbsorbtionDataset(
            os.path.join(args.experimental_root_dir, 'training'),
            stats=stats, fold=int(args.fold), train=False, augment=False,
            use_all_data=False, experimental_data=True, shuffle=False,
            X_transform=transforms_dict['x_transform'], 
            Y_transform=transforms_dict['mu_a_transform'], 
            fluence_transform=transforms_dict['fluence_transform'],
            mask_transform=transforms_dict['mask_transform']
        ),
        'test' : e2eQPATReconstructAbsorbtionDataset(
            os.path.join(args.experimental_root_dir, 'test'),
            stats=stats, fold=int(args.fold), train=False, augment=False,
            use_all_data=True, experimental_data=True, shuffle=False,
            X_transform=transforms_dict['x_transform'], 
            Y_transform=transforms_dict['mu_a_transform'], 
            fluence_transform=transforms_dict['fluence_transform'],
            mask_transform=transforms_dict['mask_transform']
        ),
    }
    dataloaders = {
        'train' : DataLoader(
            datasets['train'], batch_size=args.train_batch_size, shuffle=False, num_workers=4
        ),
        # backpropagation not performed on the validation set so batch size can be larger
        'val' : DataLoader( 
            datasets['val'], batch_size=args.val_batch_size, shuffle=False, num_workers=4
        ),
        # backpropagation not performed on the test set so batch size can be larger
        'test' : DataLoader(
            datasets['test'], batch_size=args.val_batch_size, shuffle=False, num_workers=4
        )
    }
    logging.info(f'train: {len(datasets['train'])}, val: {len(datasets['val'])}, \
        test: {len(datasets["test"])}')
    return (datasets, dataloaders, transforms_dict)


def combine_datasets(args : argparse.Namespace,
                     datasets : Dict[str, Dataset]) -> tuple:
    combined_training_dataset = CombineMultipleDatasets(datasets)
    train_loader = DataLoader(
        combined_training_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4
    )
    return (combined_training_dataset, train_loader)


def create_embedding_dataloaders(args) -> tuple:
    datasets = {
        'train' : SyntheticReconstructAbsorbtionDataset(
            args.synthetic_root_dir, split='train', data_space='latent', fold=args.fold
        ),
        'val' : SyntheticReconstructAbsorbtionDataset(
            args.synthetic_root_dir, split='val', data_space='latent', fold=args.fold
        ),
        'test' : SyntheticReconstructAbsorbtionDataset(
            args.synthetic_root_dir, split='test', data_space='latent', fold=args.fold
        )
    }
    dataloaders = {
        'train' : DataLoader(
            datasets['train'], batch_size=args.train_batch_size, shuffle=False, num_workers=4
        ),
        # backpropagation not performed on the validation set so batch size can be larger
        'val' : DataLoader( 
            datasets['val'], batch_size=args.val_batch_size, shuffle=False, num_workers=4
        ),
        # backpropagation not performed on the validation set so batch size can be larger
        'test' : DataLoader(
            datasets['test'], batch_size=args.val_batch_size, shuffle=False, num_workers=4
        )
    }
    return (datasets, dataloaders)


def save_embeddings(encode_func : Callable[[torch.Tensor], list[torch.Tensor]],
                    datasets : dict,
                    dataloaders : dict,
                    dirpath : str,
                    device : torch.device) -> None:
    """
    Save embeddings generated by an encoding function for data from dataloaders to an HDF5 file.

    Args:
        encode_func (Callable[[torch.Tensor], list[torch.Tensor]]): A function that takes a tensor and returns its embeddings.
        datasets (dict): A dictionary of datasets to load the data to be encoded.
        dataloaders (dict): A dictionary of dataloaders to load the data to be encoded.
        dirpath (str): The directory path where the embeddings file will be saved.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
    Returns:
        None
    """
    embeddings_name = os.path.join(dirpath, 'embeddings.h5')
    if os.path.exists(embeddings_name):
        logging.info(f'{embeddings_name} already exists, will be overwritten.')
    with h5py.File(embeddings_name, 'w') as f:
        for key in dataloaders.keys():
            f.create_group(key)
            for i, batch in enumerate(dataloaders[key]):
                batch_size = dataloaders[key].batch_size
                with torch.no_grad():
                    (X, Y) = batch
                    X = X.to(device)
                    Y = Y.to(device)
                    X_embeddings = encode_func(X)[0].cpu().numpy()
                    Y_embeddings = encode_func(Y)[0].cpu().numpy()
                for j in range(len(X_embeddings)):
                    sample_name = datasets[key].samples[i*batch_size+j]
                    f[key].create_group(sample_name)
                    f[key][sample_name].create_dataset(
                        'X', data=X_embeddings[j], dtype='float32'
                    )
                    f[key][sample_name].create_dataset(
                        'Y', data=Y_embeddings[j], dtype='float32'
                    )
                    
                    
def plot_test_examples(dataset : ReconstructAbsorbtionDataset, 
                       dirpath : str,
                       args : argparse.Namespace,
                       X : torch.Tensor, 
                       Y : torch.Tensor,
                       Y_hat : torch.Tensor,
                       X_hat : torch.Tensor=None, # for VAEs
                       mask : torch.Tensor=None,
                       X_transform=None, 
                       Y_transform=None,
                       X_cbar_unit : str=r'Pa J$^{-1}$',
                       Y_cbar_unit : str=r'cm$^{-1}$',
                       fig_titles : Union[tuple[str], list[str]]=None,
                       **kwargs) -> None:
    if type(X_hat) != torch.Tensor:
        X_hat = [None]*len(X)
    if type(mask) != torch.Tensor:
        mask = [None]*len(X)
    assert len(X) == len(Y) == len(Y_hat) == len(X_hat) == len(mask), 'Input tensors must \
        have the same length.'
    if not fig_titles:
        fig_titles = [f'Example {i}' for i in range(len(X))]
        
    for i in range(len(X)):
        (fig, _) = dataset.plot_comparison(
            X[i], Y[i], Y_hat[i], X_hat=X_hat[i], mask=mask[i],
            X_transform=X_transform, Y_transform=Y_transform,
            X_cbar_unit=X_cbar_unit, Y_cbar_unit=Y_cbar_unit, **kwargs
        )
        if args.wandb_log:
            wandb.log({fig_titles[i]: wandb.Image(fig)})
        if args.save_dir:
            fig.savefig(os.path.join(dirpath, fig_titles[i]+'.png'))