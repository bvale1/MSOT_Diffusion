import wandb
import json
import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from utility_classes import *

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
            
            
def create_dataloaders(args, model_name) -> tuple:
    if args.wandb_log:
        wandb.login()
        wandb.init(
            project='MSOT_Diffusion', name=model_name, 
            save_code=True, reinit=True
        )
        wandb.config = vars(args)
    
    with open(os.path.join(args.root_dir, 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    normalise_x = MaxMinNormalise(
        torch.Tensor([config['normalisation_X']['max']]),
        torch.Tensor([config['normalisation_X']['min']])
    )
    x_transform = transforms.Compose([
        ReplaceNaNWithZero(), 
        normalise_x
    ])
    normalise_y = MaxMinNormalise(
        torch.Tensor([config['normalisation_mu_a']['max']]),
        torch.Tensor([config['normalisation_mu_a']['min']])
    )
    y_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        normalise_y
    ])
        
    dataset = ReconstructAbsorbtionDataset(
        args.root_dir, gt_type='mu_a', X_transform=x_transform, Y_transform=y_transform
    )
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # train/val/test sets are always the same
    )
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
    )
    
    return (dataset, train_dataset, val_dataset, test_dataset, 
            train_loader, val_loader, test_loader, normalise_x, normalise_y)