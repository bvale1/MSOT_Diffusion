import argparse
import wandb
import logging 
import torch
import os
import json
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

import end_to_end_phantom_QPAT.utils.networks as e2eQPAT_networks
import utility_classes as uc
import utility_functions as uf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/20241208_ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs, set to zero for testing')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='batch size for inference, 4x train_batch_size should have similar device memory requirements')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--save_test_examples', help='save test examples to save_dir and wandb', action='store_true', default=False)
    parser.add_argument('--wandb_log', help='use wandb logging', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='Unet_checkpoints', help='path to save the model')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='path to load a model checkpoint')
    parser.add_argument('--early_stop_patience', type=int, default=np.inf, help='early stopping patience')
    parser.add_arguemnt('--model', choices=['UNet_smp', 'UNet_e2eQPAT'], default='UNet_smp', help='model to train')
    
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
    
    (datasets, dataloaders, normalise_x, normalise_y) = uf.create_dataloaders(
        args, args.model
    )
    
    # ==================== Model ====================
    image_size = (datasets['train'][0][0].shape[-2],  datasets['train'][0][0].shape[-1])
    channels = datasets['train'][0][0].shape[-3]
    match args.model:
        case 'UNet_smp':
            model = smp.Unet(
                encoder_name='resnet101', encoder_weights='imagenet',
                decoder_attention_type='scse', # @article{roy2018recalibrating, title={Recalibrating fully convolutional networks with spatial and channel “squeeze and excitation” blocks}, author={Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian}, journal={IEEE transactions on medical imaging}, volume={38}, number={2}, pages={540--549}, year={2018}, publisher={IEEE}}
                in_channels=channels, classes=1, 
            )
        case 'UNet_e2eQPAT':
            model = e2eQPAT_networks.RegressionUNet(
                in_channels=1, out_channels=1,
                initial_filter_size=64, kernal_size=3
            )
    
    if args.load_checkpoint:
        try:
            model.load_state_dict(torch.load(args.load_checkpoint, weights_only=True))
            logging.info(f'loaded checkpoint: {args.load_checkpoint}')
        except Exception as e:
            logging.error(f'could not load checkpoint: {args.load_checkpoint}')
    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    model.to(device)
    
    # ==================== Optimizer, lr Scheduler, Objective, Checkpointer ====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-3, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs*len(dataloaders['train']), eta_min=1e-6
    )
    mse_loss = nn.MSELoss(reduction='none')
    if args.save_dir:
        checkpointer = uc.CheckpointSaver(args.save_dir)
        with open(os.path.join(checkpointer.dirpath, 'args.json'), 'w') as f:
            json.dump(var_args, f, indent=4)
    
    
    # ==================== Training ====================
    # if mean val loss does not decrease after this many epochs, stop training
    early_stop_patience = args.early_stop_patience
    stop_counter = 0
    prev_val_loss = np.inf
    for epoch in range(args.epochs):
        # ==================== Train epoch ====================
        model.train()
        total_train_loss = 0
        best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                   'worst' : {'index' : 0, 'loss' : -np.Inf}}
        for i, (X, Y) in enumerate(dataloaders['train']):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = mse_loss(Y_hat, Y).mean(dim=(1, 2, 3))
            best_and_worst_examples = uf.get_best_and_worst(
                loss, best_and_worst_examples, i*args.train_batch_size
            )
            loss = loss.mean()
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if args.wandb_log:
                wandb.log({'train_loss' : loss.item()})
        logging.info(f'train_epoch: {epoch}, mean_train_loss: {total_train_loss/len(dataloaders['train'])}')
        logging.info(f'train_epoch {best_and_worst_examples}')
        
        # ==================== Validation epoch ====================
        model.eval()
        total_val_loss = 0
        best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                   'worst' : {'index' : 0, 'loss' : -np.Inf}}
        with torch.no_grad():
            for i, (X, Y) in enumerate(dataloaders['val']):
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)
                loss = mse_loss(Y_hat, Y).mean(dim=(1, 2, 3))
                best_and_worst_examples = uf.get_best_and_worst(
                    loss, best_and_worst_examples, i*args.val_batch_size
                )
                loss = loss.mean()
                total_val_loss += loss.item()
                if args.wandb_log:
                    wandb.log({'val_loss' : loss.item()})
        total_val_loss /= len(dataloaders['val'])
        if args.save_dir: # save model checkpoint if validation loss is lower
            checkpointer(model, epoch, total_val_loss)
        logging.info(f'val_epoch: {epoch}, mean_val_loss: {total_val_loss}')
        logging.info(f'val_epoch {best_and_worst_examples}')
        
        # check for early stopping criterion
        if total_val_loss >= prev_val_loss:
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
    test_metric_calculator = uc.TestMetricCalculator(n_samples=len(datasets['test']))
    with torch.no_grad():
        for i, (X, Y) in enumerate(dataloaders['test']):
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = model(X)
            test_metric_calculator(Y=Y, Y_hat=Y_hat)
            loss = mse_loss(Y_hat, Y).mean(dim=(1, 2, 3))
            best_and_worst_examples = uf.get_best_and_worst(
                loss, best_and_worst_examples, i
            )
            loss = loss.mean()
            total_test_loss += loss.item()
            if args.wandb_log:
                wandb.log({'test_loss' : loss.item()})
    logging.info(f'total_test_loss: {total_test_loss/len(dataloaders['test'])}')
    logging.info(f'test_epoch {best_and_worst_examples}')
    logging.info(f'test_metrics: {test_metric_calculator.get_metrics()}')
    if args.wandb_log:
        wandb.log(test_metric_calculator.get_metrics())
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
            Y_hat = model.forward(X)
        uf.plot_test_examples(
            datasets['test'], checkpointer.dirpath, args, X, Y, Y_hat,
            X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'cm$^{-1}$',
            fig_titles=['test_example0', 'test_example_best', 'test_example_worst']
        )
        
    if args.wandb_log:
        wandb.finish()