import argparse
import wandb
import logging 
import torch
import os
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import denoising_diffusion_pytorch as ddp

from vq_vae.vq_vae import VQVAE
import utility_classes as uc
import utility_functions as uf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/20241208_ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs, set to zero for testing')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='batch size for inference, approximately 4*train_batch_size has about the same vram footprint')
    parser.add_argument('--image_size', type=int, default=256, help='image size')   
    parser.add_argument('--save_test_examples', default=False, help='save test examples to save_dir and wandb', action='store_true')
    parser.add_argument('--wandb_log', default=False, help='use wandb logging', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--objective', choices=['pred_v', 'pred_noise'], default='pred_v', help='objective of the diffusion model')
    parser.add_argument('--self_condition', default=False, help='condition on the previous timestep', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='DDPM_checkpoints', help='path to save the model')
    parser.add_argument('--load_checkpoint_dir', type=str, default=None, help='path to a model checkpoint to load')
    parser.add_argument('--use_autoencoder_dir', type=str, default=None, help='path to autoencoder model')

    args = parser.parse_args()
    var_args = vars(args)
    logging.info(f'args dict: {var_args}')

    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(False)
    logging.info(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    logging.info(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    
    if args.seed:
        seed = args.seed
    else:
        seed = np.random.randint(0, 2**32 - 1)
        var_args['seed'] = seed
    logging.info(f'seed: {seed}')
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    # ==================== Data ====================

    if args.use_autoencoder_dir:
        (datasets, dataloaders) = uf.create_embedding_dataloaders(args)
        (image_datasets, image_dataloaders, normalise_x, normalise_y) = uf.create_dataloaders(
            args=args, model_name='latent_DDIM'
        )
    else:
        (datasets, dataloaders, normalise_x, normalise_y) = uf.create_dataloaders(
            args=args, model_name='DDIM'
        )
    
    # ==================== Model ====================
    input_size = (datasets['train'][0][0].shape[-2], datasets['train'][0][0].shape[-1])
    channels = datasets['train'][0][0].shape[-3]
    model = ddp.Unet(
        dim=32, channels=channels, self_condition=args.self_condition,
        image_condition=True, full_attn=False, flash_attn=False
    )
    diffusion = ddp.GaussianDiffusion(
        # objecive='pred_v' predicts the velocity field, objective='pred_noise' predicts the noise
        model, image_size=input_size, timesteps=1000,
        sampling_timesteps=100, objective=args.objective, auto_normalize=False
    )
    if args.load_checkpoint_dir:
        try:
            model.load_state_dict(torch.load(args.load_checkpoint_dir, weights_only=True))
            logging.info(f'loaded checkpoint: {args.load_checkpoint_dir}')
        except Exception as e:
            logging.error(f'could not load checkpoint: {e}')
    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of diffusion model parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    model = model.to(device)
    diffusion = diffusion.to(device)
    
    if args.use_autoencoder_dir:
        image_size = (image_datasets['train'][0][0].shape[-2],
                      image_datasets['train'][0][0].shape[-1])
        image_channels = image_datasets['train'][0][0].shape[-3]
        logging.info(f'loading autoencoder from {args.use_autoencoder_dir}')
        vqvae = VQVAE(
            in_channels=image_channels, embedding_dim=channels, num_embeddings=512,
            beta=0.25, img_size=image_size[0]
        )
        vqvae.load_state_dict(torch.load(args.use_autoencoder_dir, weights_only=True))
        vqvae = vqvae.to(device)
        vqvae.eval()
    
    # ==================== Optimizer ====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs*len(dataloaders['train']), eta_min=1e-6
    )
    if args.save_dir:
        checkpointer = uc.CheckpointSaver(args.save_dir)
        with open(os.path.join(checkpointer.dirpath, 'args.json'), 'w') as f:
            json.dump(var_args, f)
    
    # ==================== Training ====================
    early_stop_patience = 3 # if mean val loss does not decrease after this many epochs, stop training
    stop_counter = 0
    prev_val_loss = np.inf
    for epoch in range(args.epochs):
        total_train_loss = 0
        # ==================== Train epoch ====================
        model.train()
        for i, (X, Y) in enumerate(dataloaders['train']):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            # the objective is to generate Y from Gaussian noise, conditioned on X
            loss = diffusion.forward(Y, x_cond=X)
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if args.wandb_log:
                wandb.log(
                    {'train_loss' : loss.item()}
                )
        logging.info(f'train_epoch: {epoch}, mean_train_loss: {total_train_loss/len(dataloaders['train'])}')
        
        # ==================== Validation epoch ====================
        image_val_iter = iter(image_dataloaders['val'])
        if (epoch+1) % 10 == 0: # validate every 10 epochs
            model.eval()
            total_val_loss = 0
            total_val_loss_rec = 0
            best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                       'worst' : {'index' : 0, 'loss' : -np.Inf}}
            with torch.no_grad():
                for i, (X, Y) in enumerate(dataloaders['val']):
                    X = X.to(device)
                    Y = Y.to(device)
                    try:
                        Y_hat = diffusion.sample(
                            batch_size=X.shape[0], x_cond=X
                        )
                    except Exception as e:
                        logging.error(f'could not sample from diffusion model: {e}')
                        breakpoint()
                    loss = F.mse_loss(Y_hat, Y, reduction='none').mean(dim=(1, 2, 3))
                    best_and_worst_examples = uf.get_best_and_worst(
                        loss, best_and_worst_examples, i*args.val_batch_size
                    )
                    loss = loss.mean()
                    total_val_loss += loss.item()
                    if args.wandb_log:
                        wandb.log({'val_loss' : loss.item()})
                        if args.use_autoencoder_dir:
                            Y_image = next(image_val_iter)[1].to(device)
                            loss_rec = F.mse_loss(vqvae.decode(Y_hat), Y_image)
                            total_val_loss_rec += loss_rec.item()
                            wandb.log({'val_loss_rec' : loss_rec.item()})
                       
            total_val_loss /= len(dataloaders['val'])
            if args.save_dir: # save model checkpoint if validation loss is lower
                checkpointer(model, epoch, total_val_loss)
            logging.info(f'val_epoch: {epoch}, mean_val_loss: {total_val_loss}')
            if args.use_autoencoder_dir:
                logging.info(f'mean_val_loss_rec: {total_val_loss_rec/len(dataloaders['val'])}')
            logging.info(f'val_epoch {best_and_worst_examples}')
            
            # check for early stopping criterion
            if total_val_loss > prev_val_loss:
                stop_counter += 1
            else:
                stop_counter = 0
                
            if stop_counter >= early_stop_patience:
                logging.info(f'early stopping at epoch: {epoch}')
                break
                
            prev_val_loss = total_val_loss
    
    # ==================== Testing ====================
    logging.info('loading checkpoint with best validation loss for testing')
    checkpointer.load_best_model(model)
    model.eval()
    total_test_loss = 0
    total_test_loss_rec = 0
    best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                               'worst' : {'index' : 0, 'loss' : -np.Inf}}
    image_test_iter = iter(image_dataloaders['test'])
    with torch.no_grad():
        for i, (X, Y) in enumerate(dataloaders['test']):
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
            loss = F.mse_loss(Y_hat, Y, reduction='none').mean(dim=(1, 2, 3))
            best_and_worst_examples = uf.get_best_and_worst(
                loss, best_and_worst_examples, i
            )
            loss = loss.mean()
            total_test_loss += loss.item()
            if args.wandb_log:
                wandb.log({'test_loss' : loss.item()})
                if args.use_autoencoder_dir:
                    Y_image = next(image_test_iter)[1].to(device)
                    loss_rec = F.mse_loss(vqvae.decode(Y_hat), Y_image)
                    total_test_loss_rec += loss_rec.item()
                    wandb.log({'test_loss_rec' : loss_rec.item()})
    logging.info(f'mean_test_loss: {total_test_loss/len(dataloaders['test'])}')
    if args.use_autoencoder_dir:
        logging.info(f'mean_test_loss_rec: {total_test_loss_rec/len(dataloaders['test'])}')
    logging.info(f'test_epoch {best_and_worst_examples}')
    if args.save_dir and args.epochs > 0:
        torch.save(
            model.state_dict(), 
            os.path.join(
                checkpointer.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt'
            )
        )
    
    # tracking and visualising best and worst examples can highlight model 
    # failier cases, or outliers in the dataset
    if args.save_test_examples:
        model.eval()
        (X_0, Y_0) = datasets['test'][0]
        (X_best, Y_best) = datasets['test'][best_and_worst_examples['best']['index']]
        (X_worst, Y_worst) = datasets['test'][best_and_worst_examples['worst']['index']]
        X = torch.stack((X_0, X_best, X_worst), dim=0).to(device)
        Y = torch.stack((Y_0, Y_best, Y_worst), dim=0).to(device)
        with torch.no_grad():
            Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
        if args.use_autoencoder_dir:
            with torch.no_grad():
                Y_hat = vqvae.decode(Y_hat)
            (X_0, Y_0) = image_datasets['test'][0]
            (X_best, Y_best) = image_datasets['test'][best_and_worst_examples['best']['index']]
            (X_worst, Y_worst) = image_datasets['test'][best_and_worst_examples['worst']['index']]
            X = torch.stack((X_0, X_best, X_worst), dim=0).to(device)
            Y = torch.stack((Y_0, Y_best, Y_worst), dim=0).to(device)            
        uf.plot_test_examples(
            datasets['test'], checkpointer.dirpath, args, X, Y, Y_hat,
            X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'cm$^{-1}$',
            fig_titles=['test_example0', 'test_example_best', 'test_example_worst']
        )
        
    if args.wandb_log:
        wandb.finish()
    