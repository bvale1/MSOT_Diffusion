import argparse
import wandb
import logging 
import torch
import os
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
    parser.add_argument('--image_size', type=int, default=64, help='image size')   
    parser.add_argument('--save_test_examples', default=False, help='save test examples to save_dir and wandb', action='store_true')
    parser.add_argument('--wandb_log', default=False, help='use wandb logging', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='DDPM_checkpoints', help='path to save the model')
    parser.add_argument('--load_checkpoint_dir', type=str, default=None, help='path to a model checkpoint to load')
    parser.add_argument('--use_autoencoder_dir', type=str, default=None, help='path to autoencoder model')

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
    
    (datasets, dataloaders, normalise_x, normalise_y) = uf.create_dataloaders(
        args=args, model_name='DDPM'
    )
    if args.use_autoencoder_dir:
        (emb_datasets, emb_dataloaders) = uf.create_embedding_dataloaders(args)
        train_loader = emb_dataloaders['train']
        val_loader = emb_dataloaders['val']
        test_loader = emb_dataloaders['test']
    else:
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
    
    # ==================== Model ====================
    image_size = (datasets['train'].__getitem__(0)[0].shape[-2],
                  datasets['train'].__getitem__(0)[0].shape[-1])
    channels = datasets['train'].__getitem__(0)[0].shape[-3]
    model = ddp.Unet(
        dim=32, channels=channels, self_condition=True, image_condition=True,
        full_attn=False, flash_attn=False
    )
    diffusion = ddp.GaussianDiffusion(
        # objecive='pred_v' predicts the velocity field, objective='pred_noise' predicts the noise
        model, image_size=image_size, timesteps=args.epochs*len(dataloaders['train']),
        sampling_timesteps=100, objective='pred_noise', auto_normalize=False
    )
    if args.load_checkpoint_dir:
        try:
            model.load_state_dict(torch.load(args.load_checkpoint_dir))
            logging.info(f'loaded checkpoint: {args.load_checkpoint_dir}')
        except Exception as e:
            logging.error(f'could not load checkpoint: {e}')
    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of diffusion model parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    model = model.to(device)
    diffusion = diffusion.to(device)
    
    if args.use_autoencoder_dir:
        logging.info(f'loading autoencoder from {args.use_autoencoder_dir}')
        vqvae = VQVAE(
            in_channels=channels, embedding_dim=16, num_embeddings=512,
            beta=0.25, img_size=image_size[0]
        )
        vqvae.load_state_dict(torch.load(args.use_autoencoder_dir))
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
    
    # ==================== Training ====================
    early_stop_patience = 3 # if mean val loss does not decrease after this many epochs, stop training
    stop_counter = 0
    prev_val_loss = np.inf
    for epoch in range(args.epochs):
        total_train_loss = 0
        # ==================== Train epoch ====================
        model.train()
        for i, batch in enumerate(train_loader):
            (X, Y) = batch
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
        logging.info(f'train_epoch: {epoch}, mean_train_loss: {total_train_loss/len(train_loader)}')
        
        # ==================== Validation epoch ====================
        if (epoch+1) % 10 == 0: # validate every 10 epochs
            model.eval()
            total_val_loss = 0
            total_val_loss_rec = 0
            best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                       'worst' : {'index' : 0, 'loss' : -np.Inf}}
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i < 7:
                        continue
                    print(f'val batch {i+1}/{len(val_loader)}')
                    (X, Y) = batch
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
                            Y_image = dataloaders['val'].dataset.__getitem__(i)[1].to(device)
                            loss_rec = F.mse_loss(vqvae.decode(Y_hat), Y_image)
                            total_val_loss_rec += loss_rec.item()
                            wandb.log({'val_loss_rec' : loss_rec.item()})
                       
            total_val_loss /= len(val_loader)
            if args.save_dir: # save model checkpoint if validation loss is lower
                checkpointer(model, epoch, total_val_loss)
            logging.info(f'val_epoch: {epoch}, mean_val_loss: {total_val_loss}')
            if args.use_autoencoder_dir:
                logging.info(f'mean_val_loss_rec: {total_val_loss_rec/len(val_loader)}')
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
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            (X, Y) = batch
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
            loss = F.mse_loss(Y_hat, Y, reduction='none').mean(dim=(1, 2, 3))
            best_and_worst_examples = uf.get_best_and_worst(loss, best_and_worst_examples, i)
            loss = loss.mean()
            total_test_loss += loss.item()
            if args.wandb_log:
                wandb.log({'test_loss' : loss.item()})
                if args.use_autoencoder_dir:
                    Y_image = dataloaders['test'].dataset.__getitem__(i)[1].to(device)
                    loss_rec = F.mse_loss(vqvae.decode(Y_hat), Y_image)
                    total_test_loss_rec += loss_rec.item()
                    wandb.log({'test_loss_rec' : loss_rec.item()})
    logging.info(f'mean_test_loss: {total_test_loss/len(test_loader)}')
    if args.use_autoencoder_dir:
        logging.info(f'mean_test_loss_rec: {total_test_loss_rec/len(test_loader)}')
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
    if args.save_test_examples:
        model.eval()
        (X_0, Y_0) = datasets['test'][0]
        (X_best, Y_best) = datasets['test'][best_and_worst_examples['best']['index']]
        (X_worst, Y_worst) = datasets['test'][best_and_worst_examples['worst']['index']]
        X = torch.stack((X_0, X_best, X_worst), dim=0).to(device)
        with torch.no_grad():
            Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
        (fig_0, ax) = datasets['test'].plot_comparison(
            X_0, Y_0, Y_hat[0], X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$'
        )
        (fig_best, ax) = datasets['test'].plot_comparison(
            X_best, Y_best, Y_hat[1], X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$'
        )
        (fig, ax) = datasets['test'].plot_comparison(
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
    