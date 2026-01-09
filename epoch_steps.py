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
import edm2.dnnlib as dnnlib
import pickle
import copy
from typing import Literal
from typing import Iterable

import utility_functions as uf
from utility_classes import ReconstructAbsorbtionDataset

from edm2.training.phema import PowerFunctionEMA
from edm2.training.training_loop import EDM2Loss
from edm2.reconstruct_phema import reconstruct_phema
from edm2.reconstruct_phema import list_input_pickles
from edm2.generate_images import edm_sampler

def save_ema_pickles(
        ema : PowerFunctionEMA, 
        cur_nimg : float, 
        loss_fn : EDM2Loss, 
        save_dir : str,
    ) -> None:
    # Save network snapshot.
    ema_list = ema.get()
    ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
    for ema_net, ema_suffix in ema_list:
        data = dnnlib.EasyDict(loss_fn=loss_fn)
        data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
        fname = f'network-snapshot-{cur_nimg:08d}{ema_suffix}.pkl'
        print(f'Saving {fname} ... ', end='', flush=True)
        with open(os.path.join(save_dir, fname), 'wb') as f:
            pickle.dump(data, f)
        print('done')
        del data # conserve memory


def reconstruct_edm2_phema_from_dir(
        save_dir : str, out_std : list[float],
        save_reconstructions : bool = False, 
        delete_pkls : bool = False,
    ) -> Iterable[dnnlib.EasyDict]:
    pkls = list_input_pickles(save_dir)
    reconstructions = []
    if save_reconstructions:
        reconstruction_iterable = reconstruct_phema(in_pkls=pkls, out_std=out_std, out_dir=save_dir)
    else:
        reconstruction_iterable = reconstruct_phema(in_pkls=pkls, out_std=out_std, out_dir=None)
    for iteration in reconstruction_iterable:
        if iteration.out:
            reconstructions.extend(iteration['out'])
    if delete_pkls:
        for pkl in pkls:
            os.remove(pkl.path)
    return reconstructions


def test_epoch(args : arpgparse.Namespace,
               module : nn.Module,
               dataloader : DataLoader,
               synthetic_or_experimental : Literal['synthetic', 'experimental'],
               device : torch.device,
               transforms_dict : dict[str, callable],
               logging_prefix : str,
               num_steps : int = 16,
               plot_all_reconstructions : bool = False,
               dataset : ReconstructAbsorbtionDataset = None) -> None:
    total_test_loss = 0
    bg_test_metric_calculator = uc.TestMetricCalculator()
    inclusion_test_metric_calculator = uc.TestMetricCalculator()
    test_start_time = timeit.default_timer()
    len_dataloader = 0 # needed because dataloader is an iterable but may not have __len__
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            len_dataloader += 1
            X = batch[0].to(device); mu_a = batch[1].to(device); 
            fluence = batch[2].to(device); wavelength_nm = batch[3].to(device)
            files = batch[6]  # added for saving test examples

            match args.model:
                case 'UNet_e2eQPAT' | 'Swin_UNet':
                    Y_hat = module(X)
                case 'UNet_wl_pos_emb':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    Y_hat = module(X, class_labels=wavelength_nm_onehot)
                case 'UNet_diffusion_ablation':
                    Y_hat = module(X)
                case 'DDIM':
                    Y_hat = module.sample(batch_size=X.shape[0], x_cond=X)
                case 'DiT':
                    Y_hat = module.sample(
                        batch_size=X.shape[0], 
                        x_cond=X,
                        wavelength_cond=wavelength_nm.squeeze()
                    )
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
                    Y_hat = edm_sampler(
                        module, noise, x_cond=X, labels=wavelength_nm_onehot, num_steps=num_steps
                    )

            mu_a_hat = Y_hat[:, 0:1]            
            mu_a_loss = F.mse_loss(mu_a_hat, mu_a, reduction='mean')    

            if plot_all_reconstructions and args.save_dir:
                files = ['.'.join(files.split('/')[-1].split('.')[:-1]) for files in files]
                uf.plot_test_examples(
                    dataset, args.save_dir, args, X, mu_a, mu_a_hat,
                    mask=batch[5], X_transform=transforms_dict['normalise_x'], 
                    Y_transform=transforms_dict['normalise_mu_a'],
                    X_cbar_unit=r'Pa J$^{-1}$', Y_cbar_unit=r'cm$^{-1}$',
                    fig_titles=files
                )

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
    
    total_test_loss /= len_dataloader
    total_test_time = timeit.default_timer() - test_start_time
    logging.info(f'{logging_prefix}_time: {total_test_time}')
    logging.info(f'{logging_prefix}_time_per_batch: {total_test_time/len_dataloader}')
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
                   f'{logging_prefix}_time_per_batch' : total_test_time/len_dataloader})
        
    return total_test_loss, bg_test_metric_calculator, inclusion_test_metric_calculator