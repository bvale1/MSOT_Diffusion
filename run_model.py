import wandb
import logging 
import torch
import os
import json
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_warmup as warmup
import denoising_diffusion_pytorch as ddp
import peft

from edm2.training.networks_edm2 import Precond
from edm2.training.networks_edm2 import UNet as EDM2_UNet
from edm2.training.training_loop import EDM2Loss
from edm2.training.training_loop import learning_rate_schedule
from edm2.training.phema import PowerFunctionEMA
from edm2.generate_images import edm_sampler

import end_to_end_phantom_QPAT.utils.networks as e2eQPAT_networks
import utility_classes as uc
import utility_functions as uf
from epoch_steps import *
from nn_modules.time_conditioned_residual_unet import TimeConditionedResUNet
from nn_modules.DiT import DiT
from nn_modules.swin_unet import SwinTransformerSys

# An all purpose script for training, validating and testing the models
# to test a trained model set --epochs 0 and --load_checkpoint_dir to the path of the model checkpoint
# --objective and --self_condition are only for diffusion (DDIM)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(False)
    logging.info(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    logging.info(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    
    args, var_args = uf.get_config()

    if args.seed:
        seed = args.seed
    else:
        seed = 42
        var_args['seed'] = seed
    logging.info(f'seed: {seed}')
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if not torch.cuda.is_available():
        raise ValueError('cuda is not available')
    logging.info(f'using device: {device}')
    
    # ==================== Data ====================
    (experimental_datasets, experimental_dataloaders, experimental_transforms_dict) = uf.create_e2eQPAT_dataloaders(
        args, model_name=args.model, 
        stats_path=os.path.join(args.experimental_root_dir, 'dataset_stats.json')
    )
    (synthetic_datasets, synthetic_dataloaders, synthetic_transforms_dict) = uf.create_synthetic_dataloaders(
        args, model_name=args.model
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
        case 'UNet_e2eQPAT':
            model = e2eQPAT_networks.RegressionUNet(
                in_channels=channels, 
                out_channels=out_channels,
                initial_filter_size=64, 
                kernel_size=3
            )
        case 'UNet_wl_pos_emb':
            # model = ddp.Unet(
            #     dim=32, channels=channels, out_dim=out_channels,
            #     self_condition=False, image_condition=False, use_attn=args.attention,
            #     full_attn=False, flash_attn=False, learned_sinusoidal_cond=False, 
            # )
            #model = TimeConditionedResUNet(
            #    dim_in=channels, dim_out=out_channels, dim_first_layer=64,
            #    kernel_size=3, theta_pos_emb=10000, self_condition=False,
            #    image_condition=False
            #)
            model = EDM2_UNet(
                img_resolution=args.image_size,
                img_channels_in=channels,
                img_channels_out=out_channels,
                label_dim=1000,
                model_channels=64,
                attn_resolutions=[16, 8] if args.attention else [],
                noise_emb=False,
            )
        case 'UNet_diffusion_ablation':
            model = EDM2_UNet(
                img_resolution=args.image_size,
                img_channels_in=channels,
                img_channels_out=out_channels,
                label_dim=0,
                model_channels=64,
                attn_resolutions=[16, 8] if args.attention else [],
                noise_emb=False,
                num_blocks=1,
                #channel_mult=[1,2,4,8,16],
                channel_mult=[1,2,3,4,8],
            )
        case 'Swin_UNet':
            model = SwinTransformerSys(
                img_size=image_size[0], patch_size=4, in_chans=channels, num_classes=out_channels,
                embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=False,
                final_upsample="expand_first"
            )
            uf.remove_softmax(model)
        case 'DDIM':
            model = ddp.Unet(
                dim=32, channels=out_channels, out_dim=out_channels,
                self_condition=args.self_condition, image_condition=True, 
                image_condition_channels=channels, use_attn=args.attention,
                full_attn=False, flash_attn=False
            )
            #model = TimeConditionedResUNet(
            #    dim_in=out_channels, dim_out=out_channels, dim_first_layer=64,
            #    kernel_size=3, theta_pos_emb=10000, self_condition=args.self_condition,
            #    image_condition=True, dim_image_condition=channels
            #)
            diffusion = ddp.GaussianDiffusion(
                # objecive='pred_v' predicts the velocity field, objective='pred_noise' predicts the noise
                model, image_size=image_size, timesteps=1000,
                sampling_timesteps=100, objective=args.objective, auto_normalize=False,
            )
        case 'DiT':
            # parameters depth=12, hidden_size=384, and num_heads=6 are the same as DiT-S/8.
            # with an image size of 256 and patch size of 16, we have the 
            # same number of patches as ViT from an image is worth 16x16 words
            #if image_size[0] % 16 != 0:
            #    raise ValueError('image size must be divisible by 16 for DiT model')
            #patch_size = image_size[0] // 16
            patch_size = 4
            model = DiT(
                dim_in=out_channels, dim_out=out_channels, input_size=image_size, 
                depth=12, hidden_size=384, patch_size=patch_size, num_heads=6,
                self_condition=args.self_condition, image_condition=True
            )
            diffusion = ddp.GaussianDiffusion(
                # objecive='pred_v' predicts the velocity field, objective='pred_noise' predicts the noise
                model, image_size=image_size, timesteps=1000,
                sampling_timesteps=100, objective=args.objective, auto_normalize=False,
            )
        case 'EDM2':
            attn_resolutions = [16, 8] if args.attention else []
            label_dim = 1000 if args.wl_conditioning else 0
            in_channels = out_channels+1 # plus 1 for conditional information
            loss_fn = EDM2Loss(P_mean=-0.8, P_std=1.6, sigma_data=0.5)
            model = Precond(
                img_resolution=256, img_channels_in=in_channels, img_channels_out=out_channels,
                label_dim=label_dim, model_channels=64, attn_resolutions=attn_resolutions, 
                use_fp16=False, sigma_data=0.5
            )
            if not args.attention:
                uf.remove_attention(model.unet)
                
            # label_dim:
            #    > labels are one-hot encoded classes (1000 classes in ImageNet)
            #    > wavelength can be used for conditioning instead of class labels
            #    > wavelength should be passed to the Precond model as a size (batch_size, n_wavelengths)
            # - Work out what to use for sigma_data
            #    > sigma is used to comput the time embedding instead of timestep t
            # - Work out what model_channels does, and whether 192 is appropriate
            #    > Base multiplier for the number of channels.
            #    > keep as is for now.
            # - Learning rate scheduler parameters are heavily dependent heavily on the network capacity and dataset
            #   > For now the schedular is ommited
            #scheduler = learning_rate_schedule(
            #    cur_nimg=0, 
            #    batch_size=args.train_batch_size,
            #    ref_lr=100e-4,
            #    ref_batches=70e3, 
            #    rampup_Mimg=10
            #)


    if args.load_checkpoint_dir:
        model.load_state_dict(
            torch.load(args.load_checkpoint_dir, weights_only=True), strict=False
        )
        logging.info(f'loaded checkpoint: {args.load_checkpoint_dir}')
    
    if args.freeze_encoder:
        logging.info('freezing encoder')
        if args.model == 'UNet_e2eQPAT':
            logging.info('freezing encoder')
            model.freeze_encoder()
        else:
            for param in model.init_conv.parameters():
                param.requires_grad = False
            for param in model.downs.parameters():
                param.requires_grad = False

    if args.boft_rank > 0:
        match args.model:
            case 'UNet_e2eQPAT':
                target_modules = [
                    "0",  # Matches Conv2d at position 0 in Sequential blocks
                    "2",  # Matches Conv2d at position 2 in Sequential blocks
                ]
            case 'UNet_wl_pos_emb' | 'UNet_diffusion_ablation' | 'DDIM' | 'DiT' | 'EDM2' | 'Swin_UNet':
                raise NotImplementedError('BOFT not implemented for this model yet')
        boft_config = peft.BOFTConfig(
            boft_block_size=args.boft_rank,
            boft_n_butterfly_factor=2,
            target_modules=target_modules,
            modules_to_save=None,
            boft_dropout=0.0,
            bias="none",
        )
        model = peft.get_peft_model(model, boft_config)
        logging.info(f'BOFT applied with rank {args.boft_rank} to all modules')
        model.print_trainable_parameters()
        

    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    if args.wandb_log: 
        wandb.log({'number_of_parameters' : no_params})
    model.to(device)
    if args.model in ['DDIM', 'DiT']:
        diffusion.to(device)
    
    # ==================== Optimizer, lr Scheduler, Objective, Checkpointer ====================
    if args.model not in ['EDM2', 'unet_diffusion_ablation']:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, eps=1e-8, amsgrad=True
        )
    else:
        ema = PowerFunctionEMA(model, stds=[0.05, 0.1])
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=10, factor=0.9
    )
    if args.warmup_period > 1:
        warmup_scheduler = warmup.LinearWarmup(
            optimizer, warmup_period=args.warmup_period
        )
    if args.save_dir:
        checkpointer = uc.CheckpointSaver(args.save_dir, top_n=1)
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
    
    cur_nimg = 0 # needed for EDM2 lr scheduler and EMA update
    for epoch in range(args.epochs):
        # ==================== Train epoch ====================
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_loader):
            X = batch[0].to(device); mu_a = batch[1].to(device); 
            fluence = batch[2].to(device); wavelength_nm = batch[3].to(device)
            optimizer.zero_grad()
            
            match args.model:
                case 'UNet_e2eQPAT' | 'Swin_UNet':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    Y_hat = model(X, class_labels=wavelength_nm_onehot)
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X)
                case 'DDIM':
                    if args.predict_fluence:
                        loss = diffusion.forward(torch.cat((mu_a, fluence), dim=1), x_cond=X)
                    else:
                        loss = diffusion.forward(mu_a, x_cond=X)
                case 'DiT':
                    if args.predict_fluence:
                        loss = diffusion.forward(
                            torch.cat((mu_a, fluence), dim=1),
                            x_cond=X, 
                            wavelength_cond=wavelength_nm.squeeze()
                        )
                    else:
                        loss = diffusion.forward(
                            mu_a,
                            x_cond=X, 
                            wavelength_cond=wavelength_nm.squeeze()
                        )
                case 'EDM2':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    if args.predict_fluence:
                        loss = loss_fn(
                            model, torch.cat((mu_a, fluence), dim=1), 
                            x_cond=X, labels=wavelength_nm_onehot
                        )
                    else:
                        loss = loss_fn(
                            model, mu_a, 
                            x_cond=X, labels=wavelength_nm_onehot
                        )

            match args.model:
                case 'UNet_e2eQPAT' | 'UNet_wl_pos_emb' | 'UNet_diffusion_ablation' | 'Swin_UNet':
                    mu_a_hat = Y_hat[:, 0:1]
                    mu_a_loss = F.mse_loss(mu_a_hat, mu_a, reduction='mean')
                    if args.predict_fluence:
                        fluence_hat = Y_hat[:, 1:2]
                        fluence_loss = F.mse_loss(fluence_hat, fluence, reduction='mean')
                        loss = mu_a_loss + fluence_loss
                    else:
                        loss = mu_a_loss
                case 'DDIM' | 'DiT' | 'EDM2':
                    mu_a_loss = loss[:, 0:1].mean()
                    if args.predict_fluence:
                        fluence_loss = loss[:, 1:2].mean()
                        loss = mu_a_loss + fluence_loss
                    else:
                        loss = mu_a_loss
                        
            total_train_loss += loss.item()
            loss.backward()
            if args.model in ['EDM2', 'unet_diffusion_ablation']:
                lr = learning_rate_schedule(
                    cur_nimg=cur_nimg, batch_size=X.shape[0], ref_lr=0.01, ref_batches=70000, rampup_Mimg=0.1
                )
                for g in optimizer.param_groups:
                    g['lr'] = lr
                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            if args.model in ['EDM2', 'unet_diffusion_ablation']:
                # Update EMA and training state.
                cur_nimg += X.shape[0]
                ema.update(cur_nimg=cur_nimg, batch_size=X.shape[0])

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
        # only validate every 10 epochs for diffusion, due to the long sampling time
        if (args.model not in ['DDIM', 'DiT', 'EDM2']) or ((epoch+1) % 10 == 0):
            model.eval()
            if args.model in ['DDIM', 'DiT']:
                module = diffusion.eval()
            elif args.model in ['EDM2', 'unet_diffusion_ablation']:
                save_ema_pickles(ema, cur_nimg, loss_fn, args.save_dir)
                module = reconstruct_edm2_phema_from_dir(
                    args.save_dir, [args.phema_reconstruction_std], delete_pkls=True)[0]['net']
                module.to(device).float()
            else:
                module = model

            if args.synthetic_or_experimental == 'experimental' or args.synthetic_or_experimental == 'both':
                experimental_val_loss, _, _ = test_epoch(
                    args=args, module=module, dataloader=dataloaders['experimental']['val'], 
                    synthetic_or_experimental='experimental', device=device,
                    transforms_dict=transforms_dict['experimental'],
                    logging_prefix='experimental_val'
                )
                if args.wandb_log:
                    wandb.log({'mean_experimental_val_loss' : experimental_val_loss})
                if args.save_dir:
                    # priority is given to the validation loss of the experimental data
                    checkpointer(module, epoch, experimental_val_loss)
                if not args.no_lr_scheduler and args.model not in ['EDM2', 'unet_diffusion_ablation']:
                    scheduler.step(experimental_val_loss)

            if args.synthetic_or_experimental == 'synthetic' or args.synthetic_or_experimental == 'both':          
                synthetic_val_loss, _, _ = test_epoch(
                    args=args, module=module, dataloader=dataloaders['synthetic']['val'], 
                    synthetic_or_experimental='synthetic', device=device,
                    transforms_dict=transforms_dict['synthetic'],
                    logging_prefix='synthetic_val'
                )
                if args.wandb_log:
                    wandb.log({'mean_synthetic_val_loss' : synthetic_val_loss})

            if args.synthetic_or_experimental == 'synthetic':
                if args.save_dir: # save model checkpoint if validation loss is lower than previous best
                    checkpointer(module, epoch, synthetic_val_loss)
                if not args.no_lr_scheduler and args.model not in ['EDM2', 'unet_diffusion_ablation']:
                    scheduler.step(synthetic_val_loss)
                
        logging.info(f"lr: {optimizer.param_groups[0]['lr']}")
        if args.wandb_log:
            wandb.log({'lr' : optimizer.param_groups[0]['lr'],
                       'mean_train_loss' : total_train_loss/len(train_loader)})
        
    
    # ==================== Testing ====================
    logging.info('loading checkpoint with best validation loss for testing')
    checkpointer.load_best_model(model)
    model.eval()
    if args.model in ['DDIM', 'DiT']:
        module = diffusion.eval()
    elif args.model in ['EDM2', 'unet_diffusion_ablation']:
        save_ema_pickles(ema, cur_nimg, loss_fn, args.save_dir)
        module = reconstruct_edm2_phema_from_dir(args.save_dir, [args.phema_reconstruction_std])[0]['net']
        module.to(device).float()
    else:
        module = model
    if args.synthetic_or_experimental == 'experimental' or args.synthetic_or_experimental == 'both':
        experimental_test_loss, _, _ = test_epoch(
            args=args, module=module, dataloader=dataloaders['experimental']['test'], 
            synthetic_or_experimental='experimental', device=device, 
            transforms_dict=transforms_dict['experimental'], 
            logging_prefix='experimental_test'
        )
    if args.synthetic_or_experimental == 'synthetic' or args.synthetic_or_experimental == 'both':
        synthetic_test_loss, _, _ = test_epoch(
            args=args, module=module, dataloader=dataloaders['synthetic']['test'], 
            synthetic_or_experimental='synthetic', device=device, 
            transforms_dict=transforms_dict['synthetic'], 
            logging_prefix='synthetic_test'
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
                case 'UNet_e2eQPAT' | 'Swin_UNet':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    Y_hat = model(X, class_labels=wavelength_nm_onehot)
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X, torch.zeros(wavelength_nm.shape[0], device=device))
                case 'DDIM' | 'DiT':
                    Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
                case 'EDM2':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    channels = 2 if args.predict_fluence else 1
                    noise = torch.randn(
                        (X.shape[0], channels, args.image_size, args.image_size),
                        device=device
                    )
                    Y_hat = edm_sampler(module, noise, x_cond=X, labels=wavelength_nm_onehot)
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
        (X_0, mu_a_0, fluence0, wavelength_nm_0, mask_0, _, file_0) = examples_dataset[0][:7]
        (X_1, mu_a_1, fluence1, wavelength_nm_1, mask_1, _, file_1) = examples_dataset[1][:7]
        (X_2, mu_a_2, fluence2, wavelength_nm_2, mask_2, _, file_2) = examples_dataset[2][:7]
        (X_3, mu_a_3, fluence3, wavelength_nm_3, mask_3, _, file_3) = examples_dataset[3][:7]
        (X_4, mu_a_4, fluence5, wavelength_nm_4, mask_4, _, file_4) = examples_dataset[4][:7]
        files = [file_0, file_1, file_2, file_3, file_4]
        files = ['.'.join(file.split('/')[-1].split('.')[:-1]) for file in files]
        
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
                case 'UNet_e2eQPAT' | 'Swin_UNet':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    Y_hat = model(X, class_labels=wavelength_nm_onehot)
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X)
                case 'DDIM' | 'DiT':
                    Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
                case 'EDM2':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    channels = 2 if args.predict_fluence else 1
                    noise = torch.randn(
                        (X.shape[0], channels, args.image_size, args.image_size),
                        device=device
                    )
                    Y_hat = edm_sampler(module, noise, x_cond=X, labels=wavelength_nm_onehot)
        mu_a_hat = Y_hat[:, 0:1]
        if args.predict_fluence:
            fluence_hat = Y_hat[:, 1:2]
        uf.plot_test_examples(
            examples_dataset, checkpointer.dirpath, args, X, mu_a, mu_a_hat,
            mask=mask, X_transform=examples_transforms_dict['normalise_x'], 
            Y_transform=examples_transforms_dict['normalise_mu_a'],
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'cm$^{-1}$',
            fig_titles=files
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

