import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import logging
import timeit
import os
import argparse as arpgparse
import utility_classes as uc
from typing import Literal


def val_epoch(args : arpgparse.Namespace,
              module : nn.Module,
              dataloader : DataLoader,
              epoch : int, 
              device : torch.device,
              logging_prefix : str) -> float:
    total_val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X = batch[0].to(device); mu_a = batch[1].to(device); 
            fluence = batch[2].to(device); wavelength_nm = batch[3].to(device)

            match args.model:
                case 'UNet_e2eQPAT':
                    Y_hat = module(X)
                case 'UNet_wl_pos_emb':
                    Y_hat = module(X, wavelength_nm.squeeze())
                case 'UNet_diffusion_ablation':
                    Y_hat = module(X, torch.zeros(wavelength_nm.shape[0], device=device))
                case 'DDIM':
                    Y_hat = module.sample(batch_size=X.shape[0], x_cond=X)

            mu_a_hat = Y_hat[:, 0:1]            
            mu_a_loss = F.mse_loss(mu_a_hat, mu_a, reduction='mean')
            if args.predict_fluence:
                fluence_hat = Y_hat[:, 1:2]
                fluence_loss = F.mse_loss(fluence_hat, fluence, reduction='mean')
                loss = mu_a_loss + fluence_loss
            else:
                loss = mu_a_loss
            total_val_loss += loss.item()
            if args.wandb_log:
                wandb.log({f'{logging_prefix}_tot_loss' : loss.item(),
                           f'{logging_prefix}_mu_a_loss' : mu_a_loss.item()})
                if args.predict_fluence:
                    wandb.log({f'{logging_prefix}_fluence_loss' : fluence_loss.item()})
    
    total_val_loss /= len(dataloader)
    logging.info(f'{logging_prefix}_epoch: {epoch}, mean_{logging_prefix}_loss: {total_val_loss}')

    return total_val_loss


def test_epoch(args : arpgparse.Namespace,
               module : nn.Module,
               dataloader : DataLoader,
               synthetic_or_experimental : Literal['synthetic', 'experimental'],
               device : torch.device,
               transforms_dict : dict[str, callable],
               logging_prefix : str) -> None:
    total_test_loss = 0
    bg_test_metric_calculator = uc.TestMetricCalculator()
    inclusion_test_metric_calculator = uc.TestMetricCalculator()
    test_start_time = timeit.default_timer()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X = batch[0].to(device); mu_a = batch[1].to(device); 
            fluence = batch[2].to(device); wavelength_nm = batch[3].to(device)

            match args.model:
                case 'UNet_e2eQPAT':
                    Y_hat = module(X)
                case 'UNet_wl_pos_emb':
                    Y_hat = module(X, wavelength_nm.squeeze())
                case 'UNet_diffusion_ablation':
                    Y_hat = module(X, torch.zeros(wavelength_nm.shape[0], device=device))
                case 'DDIM':
                    Y_hat = module.sample(batch_size=X.shape[0], x_cond=X)

            mu_a_hat = Y_hat[:, 0:1]            
            mu_a_loss = F.mse_loss(mu_a_hat, mu_a, reduction='mean')            
            bg_test_metric_calculator(
                Y=mu_a, Y_hat=mu_a_hat, Y_transform=transforms_dict['normalise_mu_a'], 
                Y_mask=batch[4] # background mask
            )
            if synthetic_or_experimental == 'experimental':
                inclusion_test_metric_calculator(
                    Y=mu_a, Y_hat=mu_a_hat, Y_transform=transforms_dict['normalise_mu_a'],
                    Y_mask=batch[5] # inclusion mask
                )
            if args.predict_fluence:
                fluence_hat = Y_hat[:, 1:2]
                fluence_loss = F.mse_loss(fluence_hat, fluence, reduction='mean')
                loss = mu_a_loss + fluence_loss
            else:
                loss = mu_a_loss
            total_test_loss += loss.item()
            if args.wandb_log:
                wandb.log({f'{logging_prefix}_tot_loss' : loss.item(),
                           f'{logging_prefix}_mu_a_loss' : mu_a_loss.item()})
                if args.predict_fluence:
                    wandb.log({f'{logging_prefix}_fluence_loss' : fluence_loss.item()})
    
    total_test_loss /= len(dataloader)  
    total_test_time = timeit.default_timer() - test_start_time
    logging.info(f'{logging_prefix}_time: {total_test_time}')
    logging.info(f'{logging_prefix}_time_per_batch: {total_test_time/len(dataloader)}')
    logging.info(f'mean_{logging_prefix}_loss: {total_test_loss}')
    logging.info(f'background_{logging_prefix}_metrics: {bg_test_metric_calculator.get_metrics()}')
    if synthetic_or_experimental == 'experimental':
        logging.info(f'inclusion_{logging_prefix}_metrics: {inclusion_test_metric_calculator.get_metrics()}')
    if args.save_dir:
        bg_test_metric_calculator.save_metrics_all_test_samples(
            os.path.join(args.save_dir, 'background_test_metrics.json')
        )
        if synthetic_or_experimental == 'experimental':
            inclusion_test_metric_calculator.save_metrics_all_test_samples(
                os.path.join(args.save_dir, 'inclusion_test_metrics.json')
            )
    if args.wandb_log:
        bg_metrics_dict = bg_test_metric_calculator.get_metrics()
        bg_metrics_dict = {f'bg_{logging_prefix}_{key}': bg_metrics_dict[key] for key in bg_metrics_dict.keys()}
        wandb.log(bg_metrics_dict)
        if synthetic_or_experimental == 'experimental':
            inclusion_metrics_dict = inclusion_test_metric_calculator.get_metrics()
            inclusion_metrics_dict = {f'inclusion_{logging_prefix}_{key}': inclusion_metrics_dict[key] for key in inclusion_metrics_dict.keys()}
            wandb.log(inclusion_metrics_dict)
        wandb.log({f'{logging_prefix}_time' : total_test_time,
                   f'{logging_prefix}_time_per_batch' : total_test_time/len(dataloader)})
        
    return total_test_loss