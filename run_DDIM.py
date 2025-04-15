import argparse
import wandb
import logging 
import torch
import os
import json
import timeit
import numpy as np
import torch.nn as nn
import pytorch_warmup as warmup
import torch.nn.functional as F
import denoising_diffusion_pytorch as ddp

import utility_classes as uc
import utility_functions as uf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/20250130_ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--synthetic_or_experimental', choices=['experimental', 'synthetic'], default='synthetic', help='whether to use synthetic or experimental data')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs, set to zero for testing')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='batch size for inference, , 4x train_batch_size should have similar device memory requirements')
    parser.add_argument('--image_size', type=int, default=256, help='image size')   
    parser.add_argument('--save_test_examples', default=False, help='save test examples to save_dir and wandb', action='store_true')
    parser.add_argument('--wandb_log', default=False, help='use wandb logging', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--objective', choices=['pred_v', 'pred_noise'], default='pred_v', help='objective of the diffusion model')
    parser.add_argument('--self_condition', default=False, help='condition on the previous timestep', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='DDPM_checkpoints', help='path to save the model')
    parser.add_argument('--load_checkpoint_dir', type=str, default=None, help='path to a model checkpoint to load')
    parser.add_argument('--warmup_period', type=int, default=1, help='warmup period for the learning rate, must int be greater than 0')
    parser.add_argument('--data_normalisation', choices=['standard', 'minmax'], default='standard', help='normalisation method for the data')
    parser.add_argument('--fold', choices=['0', '1', '2', '3', '4'], default='0', help='fold for cross-validation, only used for experimental data')
    parser.add_argument('--wandb_notes', type=str, default='None', help='optional, comment for wandb')
    parser.add_argument('--predict_fluence', default=False, help='predict fluence as well as mu_a', action='store_true')
    parser.add_argument('--no_lr_scheduler', default=False, help='do not use lr scheduler', action='store_true')

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

    match args.synthetic_or_experimental:
        case 'experimental':
            (datasets, dataloaders, normalise_x, normalise_y, _) = uf.create_e2eQPAT_dataloaders(
                args, model_name='DDIM', 
                stats_path=os.path.join(args.root_dir, 'dataset_stats.json')
            )
        case 'synthetic':
            (datasets, dataloaders, normalise_x, normalise_y, _) = uf.create_synthetic_dataloaders(
                args=args, model_name='DDIM'
            )
    
    # ==================== Model ====================
    input_size = (datasets['test'][0][0].shape[-2], datasets['test'][0][0].shape[-1])
    channels = datasets['test'][0][0].shape[-3]
    if args.predict_fluence:
        channels *= 2
    model = ddp.Unet(
        dim=32, channels=channels, out_dim=channels,
        self_condition=args.self_condition, image_condition=True, 
        image_condition_channels=channels, full_attn=False, flash_attn=False
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
            logging.error(f'could not load checkpoint: {args.load_checkpoint_dir} {e}')
    print(model)
    no_params = sum(p.numel() for p in model.parameters())
    print(f'number of diffusion model parameters: {no_params}, model size: {no_params*4/(1024**2)} MB')
    if args.wandb_log: 
        wandb.log({'number_of_parameters' : no_params})
    model = model.to(device)
    diffusion = diffusion.to(device)
    
    # ==================== Optimizer ====================
    match args.synthetic_or_experimental:
        case 'experimental':
            # use exactly the same algorithm and hyperparamers as the e2eQPAT paper
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr
            )
        case 'synthetic':
            # Gives a bit better convergance than default Adam in my experience
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, eps=1e-3, amsgrad=True
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
    for epoch in range(args.epochs):
        # ==================== Train epoch ====================
        model.train()
        total_train_loss = 0
        best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                   'worst' : {'index' : 0, 'loss' : -np.Inf}}
        for i, batch in enumerate(dataloaders['train']):
            X = batch[0].to(device); Y = batch[1].to(device)
            optimizer.zero_grad()
            # the objective is to generate Y from Gaussian noise, conditioned on X
            if args.predict_fluence:
                fluence = batch[2].to(device)
                loss = diffusion.forward(torch.cat((Y, fluence), dim=1), x_cond=X)
            else:
                loss = diffusion.forward(Y, x_cond=X)
            mu_a_loss = loss[:, 0]            
            best_and_worst_examples = uf.get_best_and_worst(
                mu_a_loss.clone().detach(), best_and_worst_examples, i*args.val_batch_size
            )
            mu_a_loss = loss.mean()
            if args.predict_fluence:
                fluence_loss = loss[:, 1].mean()
                loss = mu_a_loss + fluence_loss
                total_train_loss += loss.item()
            else:
                loss = mu_a_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.warmup_period > 1:
                with warmup_scheduler.dampening(): # step warmup scheduler
                    pass
            if args.wandb_log:
                wandb.log(
                    {'train_tot_loss' : loss.item(),
                     'train_mu_a_loss' : mu_a_loss.item()}
                )
                if args.predict_fluence:
                    wandb.log({'train_fluence_loss' : fluence_loss.item()})
        logging.info(f'train_epoch: {epoch}, mean_train_loss: {total_train_loss/len(dataloaders['train'])}')
        
        # ==================== Validation epoch ====================
        if (epoch+1) % 10 == 0: # validate every 10 epochs
            model.eval()
            total_val_loss = 0
            best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                                       'worst' : {'index' : 0, 'loss' : -np.Inf}}
            with torch.no_grad():
                for i, batch in enumerate(dataloaders['val']):
                    X = batch[0].to(device); mu_a = batch[1].to(device)
                    
                    Y_hat = diffusion.sample(
                        batch_size=X.shape[0], x_cond=X
                    )
                    mu_a_hat = Y_hat[:, 0:1]                   
                    mu_a_loss = F.mse_loss(mu_a_hat, mu_a, reduction='none').mean(dim=(1, 2, 3))
                    best_and_worst_examples = uf.get_best_and_worst(
                        mu_a_loss, best_and_worst_examples, i*args.val_batch_size
                    )
                    mu_a_loss = mu_a_loss.mean()
                    if args.predict_fluence:
                        fluence = batch[2].to(device)
                        fluence_hat = Y_hat[:, 1:2]
                        fluence_loss = F.mse_loss(fluence_hat, fluence, reduction='none').mean()
                        loss = loss + fluence_loss
                    else:
                        loss = mu_a_loss
                    total_val_loss += loss.item()
                    if args.wandb_log:
                        wandb.log({'val_tot_loss' : loss.item(),
                                   'val_mu_a_loss' : mu_a_loss.item()})
                        if args.predict_fluence:
                            wandb.log({'val_fluence_loss' : fluence_loss.item()})
                       
            total_val_loss /= len(dataloaders['val'])
            if not args.no_lr_scheduler:
                scheduler.step(total_val_loss) # lr scheduler
            if args.save_dir: # save model checkpoint if validation loss is lower
                checkpointer(model, epoch, total_val_loss)
            logging.info(f'val_epoch: {epoch}, mean_val_loss: {total_val_loss}, lr: {scheduler.get_last_lr()}')
            logging.info(f'val_epoch {best_and_worst_examples}')            
    
    # ==================== Testing ====================
    logging.info('loading checkpoint with best validation loss for testing')
    checkpointer.load_best_model(model)
    model.eval()
    total_test_loss = 0
    best_and_worst_examples = {'best' : {'index' : 0, 'loss' : np.Inf},
                               'worst' : {'index' : 0, 'loss' : -np.Inf}}
    bg_test_metric_calculator = uc.TestMetricCalculator()
    inclusion_test_metric_calculator = uc.TestMetricCalculator()
    test_start_time = timeit.default_timer()
    with torch.no_grad():
        for i, batch in enumerate(dataloaders['test']):
            X = batch[0].to(device); mu_a = batch[1].to(device)
            Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
            mu_a_hat = Y_hat[:, 0:1]
            mu_a_loss = F.mse_loss(mu_a, mu_a_hat, reduction='none').mean(dim=(1, 2, 3))
            
            bg_test_metric_calculator(
                Y=mu_a, Y_hat=mu_a_hat, Y_transform=normalise_y, Y_mask=batch[4] # background mask
            )
            if args.synthetic_or_experimental == 'experimental':
                inclusion_test_metric_calculator(
                    Y=mu_a, Y_hat=mu_a_hat, Y_transform=normalise_y, Y_mask=batch[5] # inclusion mask
                )
            best_and_worst_examples = uf.get_best_and_worst(
                mu_a_loss, best_and_worst_examples, i
            )
            mu_a_loss = mu_a_loss.mean()
            if args.predict_fluence:
                fluence = batch[2].to(device)
                fluence_hat = Y_hat[:, 1:2]
                fluence_loss = F.mse_loss(fluence, fluence_hat, reduction='none').mean()
                loss = mu_a_loss + fluence_loss
            else:
                loss = mu_a_loss
            total_test_loss += loss.item()
            if args.wandb_log:
                wandb.log({'test_tot_loss' : loss.item(),
                           'test_mu_a_loss' : mu_a_loss.item()})
                if args.predict_fluence:
                    wandb.log({'test_fluence_loss' : fluence_loss.item()})           
    total_test_time = timeit.default_timer() - test_start_time
    logging.info(f'test_time: {total_test_time}')
    logging.info(f'test_time_per_batch: {total_test_time/len(dataloaders["test"])}')
    logging.info(f'mean_test_loss: {total_test_loss/len(dataloaders['test'])}')
    logging.info(f'test_epoch {best_and_worst_examples}')
    logging.info(f'background_test_metrics: {bg_test_metric_calculator.get_metrics()}')
    if args.synthetic_or_experimental == 'experimental':
        logging.info(f'inclusion_test_metrics: {inclusion_test_metric_calculator.get_metrics()}')
    if args.save_dir:
        bg_test_metric_calculator.save_metrics_all_test_samples(
            os.path.join(args.save_dir, 'background_test_metrics.json')
        )
        if args.synthetic_or_experimental == 'experimental':
            inclusion_test_metric_calculator.save_metrics_all_test_samples(
                os.path.join(args.save_dir, 'inclusion_test_metrics.json')
            )
    if args.wandb_log:
        wandb.log(bg_test_metric_calculator.get_metrics())
        if args.synthetic_or_experimental == 'experimental':
            inclusion_metrics_dict = inclusion_test_metric_calculator.get_metrics()
            for key in inclusion_metrics_dict.keys():
                wandb.log({'inclusion_'+key : inclusion_metrics_dict[key]})
        wandb.log({'test_time' : total_test_time,
                   'test_time_per_batch' : total_test_time/len(dataloaders['test'])})
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
        (X_0, mu_a_0, _, wavelength_nm_0, mask_0) = datasets['test'][0][:5]
        (X_1, mu_a_1, _, wavelength_nm_1, mask_1) = datasets['test'][1][:5]
        (X_2, mu_a_2, _, wavelength_nm_2, mask_2) = datasets['test'][2][:5]
        (X_best, mu_a_best, _, wavelength_nm_best, mask_best) = datasets['test'][best_and_worst_examples['best']['index']][:5]
        (X_worst, mu_a_worst, _, wavelength_nm_worst, mask_worst) = datasets['test'][best_and_worst_examples['worst']['index']][:5]
        X = torch.stack((X_0, X_1, X_2, X_best, X_worst), dim=0).to(device)
        mu_a = torch.stack((mu_a_0, mu_a_1, mu_a_2, mu_a_best, mu_a_worst), dim=0).to(device)
        mask = torch.stack((mask_0, mask_1, mask_2, mask_best, mask_worst), dim=0)
        wavelength_nm = torch.stack(
            (wavelength_nm_0, wavelength_nm_1,  wavelength_nm_2,
             wavelength_nm_best, wavelength_nm_worst), dim=0
        )
        with torch.no_grad():
            Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
            mu_a_hat = Y_hat[:, 0:1]
            #fluence_hat = Y_hat[:, 1:2]
        uf.plot_test_examples(
            datasets['test'], checkpointer.dirpath, args, X, mu_a, mu_a_hat,
            mask=mask, X_transform=normalise_x, Y_transform=normalise_y,
            X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'cm$^{-1}$',
            fig_titles=['test_example0', 'test_example1', 'test_example2',
                        'test_example_best', 'test_example_worst']
        )
        
    if args.wandb_log:
        wandb.finish()
    