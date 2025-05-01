import argparse
import wandb
import logging 
import torch
import os
import json
import h5py
import timeit
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_warmup as warmup
import segmentation_models_pytorch as smp
import denoising_diffusion_pytorch as ddp

import end_to_end_phantom_QPAT.utils.networks as e2eQPAT_networks
import utility_classes as uc
import utility_functions as uf
from UNet_training_steps import UNet_val_epoch, UNet_test_epoch

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_or_experimental', choices=['experimental', 'synthetic', 'both'], default='synthetic', help='whether to use synthetic or experimental data')
    parser.add_argument('--experimental_root_dir', type=str, default='/mnt/e/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms/', help='path to the root directory of the experimental dataset.')
    parser.add_argument('--synthetic_root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/20250327_ImageNet_MSOT_Dataset/', help='path to the root directory of the synthetic dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs, set to zero for testing')
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
    parser.add_argument('--model', choices=['UNet_smp', 'UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation'], default='UNet_smp', help='model to train')
    parser.add_argument('--data_normalisation', choices=['standard', 'minmax'], default='standard', help='normalisation method for the data')
    parser.add_argument('--fold', choices=['0', '1', '2', '3', '4'], default='0', help='fold for cross-validation, only used for experimental data')
    parser.add_argument('--wandb_notes', type=str, default='None', help='optional, comment for wandb')
    parser.add_argument('--predict_fluence', default=False, help='predict fluence as well as mu_a', action='store_true')
    parser.add_argument('--no_lr_scheduler', default=False, help='do not use lr scheduler', action='store_true')
    parser.add_argument('--freeze_encoder', default=False, help='freeze the encoder', action='store_true')
    
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
    if not torch.cuda.is_available():
        raise ValueError('cuda is not available')
    logging.info(f'using device: {device}')
    
    # ==================== Data ====================
    (experimental_datasets, experimental_dataloaders, experimental_transforms_dict) = uf.create_e2eQPAT_dataloaders(
        args, args.model, 
        stats_path=os.path.join(args.experimental_root_dir, 'dataset_stats.json')
    )
    (synthetic_datasets, synthetic_dataloaders, synthetic_transforms_dict) = uf.create_synthetic_dataloaders(
        args, args.model
    )
    datasets = {'synthetic' : synthetic_datasets, 'experimental' : experimental_datasets}
    dataloaders = {'synthetic' : synthetic_dataloaders, 'experimental' : experimental_dataloaders}
    transforms_dict = {'synthetic' : synthetic_transforms_dict, 'experimental' : experimental_transforms_dict}
    if args.synthetic_or_experimental == 'both':
        combined_training_dataset, train_loader = uf.combine_datasets(
            args, {'synthetic' : synthetic_datasets['train'], 'experimental' : experimental_datasets['train']}
        )
        datasets['combined'] = {'train' : combined_training_dataset}
        dataloaders['combined'] = {'train' : train_loader}
    # ==================== Model ====================
    image_size = (args.image_size, args.image_size)
    channels = datasets['synthetic']['test'][0][0].shape[-3]
    out_channels = channels * 2 if args.predict_fluence else channels
    match args.model:
        case 'UNet_smp':
            model = smp.Unet(
                encoder_name='resnet101', encoder_weights='imagenet',
                decoder_attention_type='scse', # @article{roy2018recalibrating, title={Recalibrating fully convolutional networks with spatial and channel “squeeze and excitation” blocks}, author={Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian}, journal={IEEE transactions on medical imaging}, volume={38}, number={2}, pages={540--549}, year={2018}, publisher={IEEE}}
                in_channels=channels, classes=out_channels
            )
            uf.reset_weights(model)
        case 'UNet_e2eQPAT':
            model = e2eQPAT_networks.RegressionUNet(
                in_channels=channels, out_channels=out_channels,
                initial_filter_size=64, kernel_size=3
            )
        case 'UNet_wl_pos_emb' | 'UNet_diffusion_ablation':
            model = ddp.Unet(
                dim=32, channels=channels, out_dim=out_channels,
                self_condition=False, image_condition=False, full_attn=False,
                flash_attn=False, learned_sinusoidal_cond=False
            )
    
    if args.load_checkpoint_dir:
        model.load_state_dict(torch.load(args.load_checkpoint_dir, weights_only=True))
        logging.info(f'loaded checkpoint: {args.load_checkpoint_dir}')
    
    if args.freeze_encoder and args.model == 'UNet_e2eQPAT':
        logging.info('freezing encoder')
        model.freeze_encoder()

    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    if args.wandb_log: 
        wandb.log({'number_of_parameters' : no_params})
    model.to(device)
    
    # ==================== Optimizer, lr Scheduler, Objective, Checkpointer ====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-8, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=10, factor=0.9
    )
    if args.warmup_period > 1:
        warmup_scheduler = warmup.LinearWarmup(
            optimizer, warmup_period=args.warmup_period
        )
    if args.save_dir:
        checkpointer = uc.CheckpointSaver(args.save_dir)
        with open(os.path.join(checkpointer.dirpath, 'args.json'), 'w') as f:
            json.dump(var_args, f, indent=4)
    
    
    # ==================== Training ====================
    match args.synthetic_or_experimental:
        case 'synthetic':
            train_loader = dataloaders['synthetic']['train']
        case 'experimental':
            train_loader = dataloaders['experimental']['train']
        case 'both':
            train_loader = dataloaders['combined']['train']
    for epoch in range(args.epochs):
        # ==================== Train epoch ====================
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_loader):
            (X, mu_a, fluence, wavelength_nm, _) = batch[:5]
            X = X.to(device); mu_a = mu_a.to(device); 
            optimizer.zero_grad()
            
            match args.model:
                case 'UNet_smp' | 'UNet_e2eQPAT':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    Y_hat = model(X, wavelength_nm.to(device).squeeze())
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X, torch.zeros(wavelength_nm.shape[0], device=device))

            mu_a_hat = Y_hat[:, 0:1]            
            mu_a_loss = F.mse_loss(mu_a_hat, mu_a, reduction='mean')
            if args.predict_fluence:
                fluence = fluence.to(device)
                fluence_hat = Y_hat[:, 1:2]
                fluence_loss = F.mse_loss(fluence_hat, fluence, reduction='mean')
                loss = mu_a_loss + fluence_loss
            else:
                loss = mu_a_loss
            total_train_loss += loss.item()            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.warmup_period > 1:
                with warmup_scheduler.dampening(): # step warmup schedulers
                    pass
            if args.wandb_log:
                wandb.log({'train_tot_loss' : loss.item(),
                           'train_mu_a_loss' : mu_a_loss.item()})
                if args.predict_fluence:
                    wandb.log({'train_fluence_loss' : fluence_loss.item()})
        logging.info(f'train_epoch: {epoch}, mean_train_loss: {total_train_loss/len(train_loader)}')
        
        # ==================== Validation epoch ====================
        model.eval()
        if args.synthetic_or_experimental == 'experimental' or args.synthetic_or_experimental == 'both':
            experimental_val_loss = UNet_val_epoch(
                args, model, dataloaders['experimental']['val'], 
                epoch, device, 'experimental_val'
            )
            if args.wandb_log:
                wandb.log({'mean_experimental_val_loss' : experimental_val_loss})
            if args.save_dir:
                # priority is given to the validation loss of the experimental data
                checkpointer(model, epoch, experimental_val_loss)
            if not args.no_lr_scheduler:
                scheduler.step(experimental_val_loss)
        if args.synthetic_or_experimental == 'synthetic' or args.synthetic_or_experimental == 'both':          
            synthetic_val_loss = UNet_val_epoch(
                args, model, dataloaders['synthetic']['val'], 
                epoch, device, 'synthetic_val'
            )
            if args.wandb_log:
                wandb.log({'mean_synthetic_val_loss' : synthetic_val_loss})
        if args.synthetic_or_experimental == 'synthetic':
            if args.save_dir: # save model checkpoint if validation loss is lower than previous best
                checkpointer(model, epoch, synthetic_val_loss)
            if not args.no_lr_scheduler:
                scheduler.step(synthetic_val_loss)
            
        logging.info(f'lr: {scheduler.get_last_lr()[0]}')
        if args.wandb_log:
            wandb.log({'lr' : scheduler.get_last_lr()[0],
                       'mean_train_loss' : total_train_loss/len(train_loader)})
        
    
    # ==================== Testing ====================
    logging.info('loading checkpoint with best validation loss for testing')
    checkpointer.load_best_model(model)
    model.eval()
    if args.synthetic_or_experimental == 'experimental' or args.synthetic_or_experimental == 'both':
        experimental_test_loss = UNet_test_epoch(
            args, model, dataloaders['experimental']['test'], 
            'experimental', device, transforms_dict['experimental'], 
            'experimental_test'
        )
    if args.synthetic_or_experimental == 'synthetic' or args.synthetic_or_experimental == 'both':
        synthetic_test_loss = UNet_test_epoch(
            args, model, dataloaders['synthetic']['test'], 
            'synthetic', device, transforms_dict['synthetic'], 
            'synthetic_test'
        )
    
    if args.save_dir and args.epochs > 0:
        torch.save(
            model.state_dict(), 
            os.path.join(
                checkpointer.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt'
            )
        )
        
    # to study overfitting, sample all images from the training set and calculate the loss
    # use model at test epoch with zero grad to get an unbiased estimate of the training loss
    best_checkpoint_train_mu_a_loss = 0
    match args.synthetic_or_experimental:
        case 'experimental' | 'both':
            train_loader = dataloaders['experimental']['train']
            examples_dataset = datasets['experimental']['test']
            examples_transforms_dict = transforms_dict['experimental']
        case 'synthetic':
            train_loader = dataloaders['synthetic']['train']
            examples_dataset = datasets['synthetic']['test']
            examples_transforms_dict = transforms_dict['synthetic']
    with torch.no_grad():        
        for i, batch in enumerate(train_loader):
            (X, mu_a, _, wavelength_nm, _) = batch[:5]
            X = X.to(device); mu_a = mu_a.to(device)
            match args.model:
                case 'UNet_smp' | 'UNet_e2eQPAT':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    Y_hat = model(X, wavelength_nm.to(device).squeeze())
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X, torch.zeros(wavelength_nm.shape[0], device=device))
            mu_a_hat = Y_hat[:, 0:1]
            mu_a_loss = F.mse_loss(mu_a, mu_a_hat, reduction='mean')
            best_checkpoint_train_mu_a_loss += mu_a_loss.item()
    best_checkpoint_train_mu_a_loss /= len(train_loader)
    best_checkpoint_val_mu_a_loss = checkpointer.best_metric_val
    overfitting_ratio = best_checkpoint_val_mu_a_loss / best_checkpoint_train_mu_a_loss
    logging.info(f'best_checkpoint_train_mu_a_loss: {best_checkpoint_train_mu_a_loss}')
    logging.info(f'best_checkpoint_val_mu_a_loss: {best_checkpoint_val_mu_a_loss}')
    logging.info(f'overfitting_ratio: {overfitting_ratio}')
    if args.wandb_log:
        wandb.log({'overfitting_ratio' : overfitting_ratio})
    
    # ==================== Save test examples ====================
    if args.save_test_examples:
        model.eval()               
        (X_0, mu_a_0, fluence0, wavelength_nm_0, mask_0) = examples_dataset[0][:5]
        (X_1, mu_a_1, fluence1, wavelength_nm_1, mask_1) = examples_dataset[1][:5]
        (X_2, mu_a_2, fluence2, wavelength_nm_2, mask_2) = examples_dataset[2][:5]
        (X_3, mu_a_3, fluence3, wavelength_nm_3, mask_3) = examples_dataset[3][:5]
        (X_4, mu_a_4, fluence5, wavelength_nm_4, mask_4) = examples_dataset[4][:5]
        
        X = torch.stack((X_0, X_1, X_2, X_3, X_4), dim=0).to(device)
        mu_a = torch.stack((mu_a_0, mu_a_1, mu_a_2, mu_a_3, mu_a_4), dim=0)
        mask = torch.stack((mask_0, mask_1, mask_2, mask_3, mask_4), dim=0)
        fluence = torch.stack((fluence0, fluence1, fluence2, fluence3, fluence5), dim=0)
        wavelength_nm = torch.stack(
            (wavelength_nm_0, wavelength_nm_1, wavelength_nm_2,
             wavelength_nm_3, wavelength_nm_4), dim=0
        ).to(device)
        with torch.no_grad():
            match args.model:
                case 'UNet_smp' | 'UNet_e2eQPAT':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    Y_hat = model(X, wavelength_nm.squeeze())
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X, torch.zeros(wavelength_nm.shape[0], device=device))
        mu_a_hat = Y_hat[:, 0:1]
        if args.predict_fluence:
            fluence_hat = Y_hat[:, 1:2]
        uf.plot_test_examples(
            examples_dataset, checkpointer.dirpath, args, X, mu_a, mu_a_hat,
            mask=mask, X_transform=examples_transforms_dict['normalise_x'], 
            Y_transform=examples_transforms_dict['normalise_mu_a'],
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'cm$^{-1}$',
            fig_titles=['test_example0', 'test_example1', 'test_example2',
                        'test_example3', 'test_example4']
        )
        if args.save_dir:
            with h5py.File(os.path.join(args.save_dir, 'test_examples.h5'), 'w') as f:
                f.create_dataset('X', data=X.cpu().numpy())
                f.create_dataset('mu_a', data=mu_a.cpu().numpy())
                f.create_dataset('mu_a_hat', data=mu_a_hat.cpu().numpy())
                f.create_dataset('fluence', data=fluence.cpu().numpy())
                if args.predict_fluence:
                    f.create_dataset('fluence_hat', data=fluence_hat.cpu().numpy())
                f.create_dataset('mask', data=mask.cpu().numpy())
                f.create_dataset('wavelength_nm', data=wavelength_nm.cpu().numpy())
    if args.wandb_log:
        wandb.finish()

