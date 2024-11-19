import argparse
import wandb
import logging 
import torch
import os
import json
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utility_classes import *
from nn_modules.AutoEncoders import VAE
from nn_modules.AutoEncoders import AutoEncoder
from nn_modules.nn_blocks import Encoder
from nn_modules.nn_blocks import Decoder
from utility_functions import remove_batchnorm

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_test_example', help='disable save test examples to wandb', action='store_false')
    parser.add_argument('--wandb_log', help='disable wandb logging', action='store_false', default=True)
    parser.add_argument('--latent_dim', type=int, default=512, help='dimension of the latent space')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='kl divergence weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_model_dir', type=str, default='AutoEncoder_checkpoints', help='path to save the model')
    parser.add_argument('--save_embeddings', action='store_true', help='save latent space embeddings of the dataset')

    args = parser.parse_args()
    logging.info(f'args dict: {vars(args)}')

    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(False)
    logging.info(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    logging.info(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    
    if args.seed:
        seed = args.seed
    else:
        seed = np.random.randint(0, 2**32 - 1)
    logging.info(f'seed: {seed}')
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    # ==================== Data ====================
    if args.wandb_log:
        wandb.login()
        wandb.init(
            project='MSOT_Diffusion', name='AutoEncoder', 
            save_code=True, reinit=True
        )
        wandb.config = vars(args)
    
    with open(os.path.join(args.root_dir, 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    normalise_x = MeanStdNormalise(
        torch.Tensor([config['normalisation_X']['mean']]),
        torch.Tensor([config['normalisation_X']['std']])
    )
    x_transform = transforms.Compose([
        ReplaceNaNWithZero(), 
        normalise_x
    ])
    
    normalise_y = MeanStdNormalise(
        torch.Tensor([config['normalisation_mu_a']['mean']]),
        torch.Tensor([config['normalisation_mu_a']['std']])
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
    
    # ==================== Model ====================
    sample_size = (dataset.__getitem__(0)[0].shape[1], dataset.__getitem__(0)[0].shape[2])
    model = AutoEncoder(
        # encoder (1,256,256) -> (32,128,128) -> (64,64,64) -> (128,32,32) -> (256,16,16) -> (512,8,8) -> (512,4,4) -> (latent_dim)
        Encoder(input_size=sample_size,
                input_channels=1,
                hidden_channels=[32, 64, 128, 256, 512, 512], 
                latent_dim=args.latent_dim),
        # decoder (latent_dim) -> (512,4,4) -> (512,8,8) -> (256,16,16) -> (128,32,32) -> (64,64,64) -> (32,128,128) -> (1,256,256)
        Decoder(hidden_channels=[512, 512, 256, 128, 64, 32],
                output_channels=1,
                output_size=sample_size,
                latent_dim=args.latent_dim)
    )
    
    remove_batchnorm(model) 
    print(model.encoder)
    print(model.decoder)
    model.to(device)
    
    # ==================== Optimizer ====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-3, amsgrad=True
    )
    mse_loss = nn.MSELoss()
    if args.save_model_dir:
        checkpointer = CheckpointSaver(args.save_model_dir)
    
    # ==================== Training ====================
    early_stop_patience = 3 # if mean val mse does not decrease after this many epochs, stop training
    stop_counter = 0
    prev_val_mse = np.inf
    for epoch in range(args.epochs):
        total_train_mse = 0
        # ==================== Train epoch ====================
        model.train()
        for i, batch in enumerate(train_loader):
            X = batch[0].to(device)
            optimizer.zero_grad()
            X_hat = model(X)
            mse = mse_loss(X_hat, X)
            total_train_mse += mse.item()
            mse.backward()
            optimizer.step()
            if args.wandb_log:
                wandb.log({'train_mse' : mse.item()})
        logging.info(f'train_epoch: {epoch}, mean_train_mse: {total_train_mse/len(train_loader)}')
        
        # ==================== Validation epoch ====================
        model.eval()
        total_val_mse = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                X = batch[0].to(device)
                X_hat = model(X)
                mse = mse_loss(X_hat, X)
                total_val_mse += mse.item()
                if args.wandb_log:
                    wandb.log({'val_mse' : mse.item()})
        total_val_mse /= len(val_loader)
        if args.save_model_dir: # save model checkpoint if validation mse is lower
            checkpointer(model, epoch, total_val_mse)
        logging.info(f'val_epoch: {epoch}, val_mse: {total_val_mse}')
        
        # check for early stopping criterion
        if total_val_mse >= prev_val_mse:
            stop_counter += 1
        else:
            stop_counter = 0
            
        if stop_counter >= early_stop_patience:
            logging.info(f'early stopping at epoch: {epoch}')
            break
            
        prev_val_mse = total_val_mse
    
    # ==================== Testing ====================
    model.eval()
    total_test_mse = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            X = batch[0].to(device)
            X_hat = model(X)
            mse = mse_loss(X_hat, X)
            total_test_mse += mse.item()
            if args.wandb_log:
                wandb.log({'test_mse' : mse.item()})
    logging.info(f'test_mse: {total_test_mse/len(test_loader)}')
    if args.save_model_dir:
        torch.save(
            model.state_dict(), 
            os.path.join(
                checkpointer.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt'
            )
        )
    
    if args.save_test_example:
        model.eval()
        (X, Y) = test_dataset[0]
        X = X.to(device)
        with torch.no_grad():
            X_hat = model.forward(X.unsqueeze(0))[0].squeeze()
        (fig, ax) = dataset.plot_comparison(
            X, X_hat, transform=normalise_x, cbar_unit=r'Pa J$^{-1}$'
        )
        if args.wandb_log:
            wandb.log({'test_example': wandb.Image(fig)})
        if args.save_model_dir:
            fig.savefig(os.path.join(checkpointer.dirpath, 'test_example1.png'))
    
    if args.wandb_log:
        wandb.finish()