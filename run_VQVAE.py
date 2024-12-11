import argparse
import wandb
import logging 
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from vq_vae.vq_vae import VQVAE
import utility_classes as uc
import utility_functions as uf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/15102024_ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs, set to zero for testing')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='batch size for inference')
    parser.add_argument('--image_size', type=int, default=256, help='size of the input images')
    parser.add_argument('--save_test_examples', help='save test examples to save_dir and wandb', action='store_true', default=False)
    parser.add_argument('--wandb_log', help='use wandb logging', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='VQVAE_checkpoints', help='path to save the model')
    parser.add_argument('--save_embeddings', help='save the embeddings to save_dir', action='store_true', default=False)
    parser.add_argument('--load_checkpoint_dir', type=str, default=None, help='path to a model checkpoint to load')

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
        args=args, model_name='VQVAE'
    )
    
    # ==================== Model ====================
    image_size = (datasets['train'].__getitem__(0)[0].shape[-2], 
                  datasets['train'].__getitem__(0)[0].shape[-1])
    channels = datasets['train'].__getitem__(0)[0].shape[-3]
    model = VQVAE(
        in_channels=channels, embedding_dim=2, num_embeddings=512,
        beta=0.25, img_size=image_size[0]
    )
    if args.load_checkpoint_dir:
        try:
            model.load_state_dict(torch.load(args.load_checkpoint_dir))
            logging.info(f'loaded checkpoint: {args.load_checkpoint_dir}')
        except Exception as e:
            logging.error(f'could not load checkpoint: {e}')
    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    model = model.to(device)
    
    # ==================== Optimizer ====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-8, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs*len(dataloaders['train']), eta_min=1e-6
    )
    if args.save_dir:
        checkpointer = uc.CheckpointSaver(args.save_dir)
    
    # ==================== Training ====================
    early_stop_patience = 5 # if mean val loss does not decrease after this many epochs, stop training
    stop_counter = 0
    prev_val_loss = np.inf
    for epoch in range(args.epochs):
        # ==================== Train epoch ====================
        model.train()
        total_train_loss = 0
        best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                   'worst' : {'index' : 0, 'loss' : -np.Inf}}        
        for i, batch in enumerate(dataloaders['train']):
            for X in batch: # model is trained to encode both input and target
                X = X.to(device)
                optimizer.zero_grad()
                recons, X, loss_vq = model.forward(X)
                loss_vae = model.loss_function(recons, X, loss_vq)
                loss = loss_vae["loss"].mean() # Overall loss
                loss_rec = loss_vae["Reconstruction_Loss"]
                loss_vq = loss_vae["VQ_Loss"].mean()
                best_and_worst_examples = uf.get_best_and_worst(
                    loss_rec, best_and_worst_examples, i*args.train_batch_size
                )
                loss_rec = loss_rec.mean()
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
            if args.wandb_log:
                wandb.log(
                    {'train_loss' : loss.item(),
                     'train_loss_rec' : loss_rec.item(),
                     'train_loss_vq' : loss_vq.item()}
                )
        logging.info(f'train_epoch: {epoch}, mean_train_loss: {total_train_loss/(2*len(dataloaders['train']))}')
        
        # ==================== Validation epoch ====================
        model.eval()
        total_val_loss = 0
        best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                   'worst' : {'index' : 0, 'loss' : -np.Inf}}
        with torch.no_grad():
            for i, batch in enumerate(dataloaders['val']):
                for X in batch:
                    X = X.to(device)
                    recons, X, loss_vq = model.forward(X)
                    loss_vae = model.loss_function(recons, X, loss_vq)
                    loss = loss_vae["loss"].mean() # Overall loss
                    loss_rec = loss_vae["Reconstruction_Loss"]
                    loss_vq = loss_vae["VQ_Loss"].mean()
                    best_and_worst_examples = uf.get_best_and_worst(
                        loss_rec, best_and_worst_examples, i*args.val_batch_size
                    )
                    loss_rec = loss_rec.mean()
                    total_val_loss += loss.item()
                if args.wandb_log:
                    wandb.log(
                        {'val_loss' : loss.item(),
                         'val_loss_rec' : loss_rec.item(),
                         'val_loss_vq' : loss_vq.item()}
                    )
        total_val_loss /= (2*len(dataloaders['val']))
        if args.save_dir: # save model checkpoint if validation loss is lower
            checkpointer(model, epoch, total_val_loss)
        logging.info(f'val_epoch: {epoch}, mean_val_loss: {total_val_loss}')
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
    best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                               'worst' : {'index' : 0, 'loss' : -np.Inf}}
    with torch.no_grad():
        for i, batch in enumerate(dataloaders['test']):
            for X in batch:
                X = X.to(device)
                recons, X, loss_vq = model.forward(X)
                loss_vae = model.loss_function(recons, X, loss_vq)
                loss = loss_vae["loss"].mean() # Overall loss
                loss_rec = loss_vae["Reconstruction_Loss"]
                loss_vq = loss_vae["VQ_Loss"].mean()
                best_and_worst_examples = uf.get_best_and_worst(
                    loss_rec, best_and_worst_examples, i*args.val_batch_size
                )
                loss_rec = loss_rec.mean()
                total_test_loss += loss.item()
            if args.wandb_log:
                wandb.log(
                    {'test_loss' : loss.item(),
                     'test_loss_rec' : loss_rec.item(),
                     'test_loss_vq' : loss_vq.item()}
                )
    logging.info(f'mean_test_loss: {total_test_loss/(2*len(dataloaders['test']))}')
    logging.info(f'test_epoch {best_and_worst_examples}')
    if args.save_dir and args.epochs > 0:
        torch.save(
            model.state_dict(), 
            os.path.join(
                checkpointer.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt'
            )
        )
    
    if args.save_embeddings:
        uf.save_embeddings(
            model.encode, datasets, dataloaders, checkpointer.dirpath, device
        )
    
    # tracking and visualising best and worst examples can highlight model 
    # deficiencies, or outliers in the dataset
    if args.save_test_examples:
        model.eval()
        (X_0, Y_0) = datasets['test'][0]
        (X_best, Y_best) = datasets['test'][best_and_worst_examples['best']['index']]
        (X_worst, Y_worst) = datasets['test'][best_and_worst_examples['worst']['index']]
        X = torch.stack((X_0, X_best, X_worst), dim=0).to(device)
        #min_X = torch.max(X.view(X.size(0),-1), dim=-1).values.to('cpu')
        #max_X = torch.max(X.view(X.size(0),-1), dim=-1).values.to('cpu')
        Y = torch.stack((Y_0, Y_best, Y_worst), dim=0).to(device)
        #min_Y = torch.min(Y.view(X.size(0),-1), dim=-1).values.to('cpu')
        #max_Y = torch.max(Y.view(X.size(0),-1), dim=-1).values.to('cpu')
        with torch.no_grad():
            X_hat = model.generate(X)
            Y_hat = model.generate(Y)
        (fig_0, ax) = datasets['main'].plot_comparison(
            X_0, Y_0, Y_hat[0], X_hat=X_hat[0],
            X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$',
            #min_X=min_X[0], max_X=max_X[0], min_Y=min_Y[0], max_Y=max_Y[0]
        )
        (fig_best, ax) = datasets['main'].plot_comparison(
            X_best, Y_best, Y_hat[1], X_hat=X_hat[1], 
            X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$',
            #min_X=min_X[1], max_X=max_X[1], min_Y=min_Y[1], max_Y=max_Y[1]
        )
        (fig, ax) = datasets['main'].plot_comparison(
            X_worst, Y_worst, Y_hat[2], X_hat=X_hat[2],
            X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'm$^{-1}$',
            #min_X=min_X[2], max_X=max_X[2], min_Y=min_Y[2], max_Y=max_Y[2]
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
    