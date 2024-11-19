import argparse
import wandb
import logging 
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import denoising_diffusion_pytorch as ddp

import utility_classes as uc
import utility_functions as uf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/15102024ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--downsample_channels_factor', type=int, default=None, help='downsample channels in each layer by this factor, use only for testing code')
    parser.add_argument('--save_test_example', help='disable save test examples to wandb', action='store_false')
    parser.add_argument('--wandb_log', help='disable wandb logging', action='store_false', default=True)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='DDPM_checkpoints', help='path to save the model')

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
    
    (dataset, train_dataset, val_dataset, test_dataset, train_loader, 
     val_loader, test_loader, normalise_x, normalise_y) = uf.create_dataloaders(args, 'DDPM')
    
    # ==================== Model ====================
    image_size = (dataset.__getitem__(0)[0].shape[1], dataset.__getitem__(0)[0].shape[2])
    model = ddp.Unet(
        dim=64, channels=1, self_condition=True, image_condition=True,
        full_attn=False, flash_attn=False,
        downsample_channels_factor=args.downsample_channels_factor
    )
    diffusion = ddp.GaussianDiffusion(
        # objecive='pred_v' predicts the velocity field, objective='pred_noise' predicts the noise
        model, image_size=image_size, timesteps=args.epochs*len(train_loader),
        objective='pred_noise'
    )
    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    model = model.to(device)
    diffusion = diffusion.to(device)
    
    # ==================== Optimizer ====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs*len(train_loader), eta_min=1e-6
    )
    if args.save_dir:
        checkpointer = uc.CheckpointSaver(args.save_dir)
    
    # ==================== Training ====================
    early_stop_patience = 3 # if mean val mse does not decrease after this many epochs, stop training
    stop_counter = 0
    prev_val_mse = np.inf
    for epoch in range(args.epochs):
        total_train_mse = 0
        # ==================== Train epoch ====================
        model.train()
        for i, batch in enumerate(train_loader):
            (X, Y) = batch
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            # the objective is to generate Y from Gaussian noise, conditioned on X
            loss = diffusion.forward(Y, x_cond=X)
            total_train_mse += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if args.wandb_log:
                wandb.log(
                    {'train_mse' : loss.item()}
                )
        logging.info(f'train_epoch: {epoch}, mean_train_mse: {total_train_mse/len(train_loader)}')
        
        # ==================== Validation epoch ====================
        if epoch % 100 == 0: # validate every 100 epochs
            model.eval()
            total_val_mse = 0
            best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                    'worst' : {'index' : 0, 'loss' : -np.Inf}}
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    (X, Y) = batch
                    X = X.to(device)
                    Y = Y.to(device)
                    Y_hat = diffusion.sample(batch_size=args.batch_size, x_cond=X)
                    mse = F.mse_loss(Y_hat, Y).mean(dim=(1, 2, 3))
                    if mse.min().item() < best_and_worst_examples['best']['loss']:
                        best_and_worst_examples['best']['loss'] = mse.min().item()
                        best_and_worst_examples['best']['index'] = i*mse.argmin().item()
                    if mse.max().item() > best_and_worst_examples['worst']['loss']:
                        best_and_worst_examples['worst']['loss'] = mse.max().item()
                        best_and_worst_examples['worst']['index'] = i*mse.argmax().item()
                    mse = mse.mean()
                    total_val_mse += mse.item()
                    if args.wandb_log:
                        wandb.log({'val_mse' : mse.item()})
            total_val_mse /= len(val_loader)
            if args.model_save_dir: # save model checkpoint if validation mse is lower
                checkpointer(model, epoch, total_val_mse)
            logging.info(f'val_epoch: {epoch}, val_mse: {total_val_mse}')
            logging.info(f'val_epoch {best_and_worst_examples}')
            
            # check for early stopping criterion
            if total_val_mse > prev_val_mse:
                stop_counter += 1
            else:
                stop_counter = 0
                
            if stop_counter >= early_stop_patience:
                logging.info(f'early stopping at epoch: {epoch}')
                break
                
            prev_val_mse = total_val_mse
    
    # ==================== Testing ====================
    logging.info('loading checkpoint with best validation mse for testing')
    checkpointer.load_best_model(model)
    model.eval()
    total_test_mse = 0
    best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                               'worst' : {'index' : 0, 'loss' : -np.Inf}}
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            (X, Y) = batch
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = diffusion.sample(batch_size=args.batch_size, x_cond=X)
            mse = F.mse_loss(Y_hat, Y).mean(dim=(1, 2, 3))
            if mse.min().item() < best_and_worst_examples['best']['loss']:
                best_and_worst_examples['best']['loss'] = mse.min().item()
                best_and_worst_examples['best']['index'] = i*mse.argmin().item()
            if mse.max().item() > best_and_worst_examples['worst']['loss']:
                best_and_worst_examples['worst']['loss'] = mse.max().item()
                best_and_worst_examples['worst']['index'] = i*mse.argmax().item()
            mse = mse.mean()
            total_test_mse += mse.item()
            if args.wandb_log:
                wandb.log(
                    {'test_mse' : mse.item()}
                )
    logging.info(f'test_mse: {total_test_mse/len(test_loader)}')
    logging.info(f'test_epoch {best_and_worst_examples}')
    if args.save_dir:
        torch.save(
            model.state_dict(), 
            os.path.join(
                checkpointer.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt'
            )
        )
    
    # tracking and visualising best and worst examples can highlight model 
    # deficiencies, or outliers in the dataset
    if args.save_test_example:
        model.eval()
        (X_0, Y_0) = test_dataset[0]
        (X_best, Y_best) = test_dataset[best_and_worst_examples['best']['index']]
        (X_worst, Y_worst) = test_dataset[best_and_worst_examples['worst']['index']]
        X = torch.stack((X_0, X_best, X_worst), dim=0).to(device)
        with torch.no_grad():
            Y_hat = diffusion.sample(batch_size=1, x_cond=X)
        (fig_0, ax) = dataset.plot_comparison(
            X_0, Y_0, Y_hat[0], X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$'
        )
        (fig_best, ax) = dataset.plot_comparison(
            X_best, Y_best, Y_hat[1], X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$'
        )
        (fig, ax) = dataset.plot_comparison(
            X_worst, Y_worst, Y_hat[2], X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$'
        )
        if args.wandb_log:
            wandb.log({'test_example0': wandb.Image(fig_0)})
            wandb.log({'test_example_best': wandb.Image(fig_best)})
            wandb.log({'test_example_worst': wandb.Image(fig)})
        if args.save_dir:
            fig_0.savefig(os.path.join(checkpointer.dirpath, 'test_example0.png'))
            fig_best.savefig(os.path.join(checkpointer.dirpath, 'test_example_best.png'))
            fig.savefig(os.path.join(checkpointer.dirpath, 'test_example_worst.png'))
        
    if args.wandb_log:
        wandb.finish()
    