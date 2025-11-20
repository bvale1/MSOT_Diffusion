import argparse
import logging 
import torch
import os
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from types import SimpleNamespace

# imports needed for unpickling edm2 package code
import edm2
import edm2.dnnlib
import edm2.generate_images
import edm2.reconstruct_phema
import edm2.training.phema 
import edm2.training.training_loop
import edm2.training.networks_edm2

import utility_functions as uf
from epoch_steps import *


def evaluate_phema_sweep(modules, args_namespace, var_args, dataloaders, transforms_dict, device):
    bg_RMSEs, bg_Rel_Errs, inclusion_RMSEs, inclusion_Rel_Errs = [], [], [], []
    for i, module in enumerate(modules):
        logging.info(f'Evaluating module {i+1}/{len(modules)}')
        module = module['net'].to(device).float()
        if var_args['synthetic_or_experimental'] == 'experimental':
            _, bg_metric_calculator, inclusion_metric_calculator = test_epoch(
                args=args_namespace, module=module, dataloader=dataloaders['experimental']['val'], 
                synthetic_or_experimental='experimental', device=device, 
                transforms_dict=transforms_dict['experimental'],
                logging_prefix='experimental_val'
            )
            inclusion_metrics = inclusion_metric_calculator.get_metrics()
            inclusion_RMSEs.append(inclusion_metrics['mean_RMSE'])
            inclusion_Rel_Errs.append(inclusion_metrics['mean_Rel_Err'])


        elif var_args['synthetic_or_experimental'] == 'synthetic':          
            _, bg_metric_calculator, _ = test_epoch(
                args=args_namespace, module=module, dataloader=dataloaders['synthetic']['val'],
                synthetic_or_experimental='synthetic', device=device, 
                transforms_dict=transforms_dict['synthetic'],
                logging_prefix='synthetic_val'
            )

        bg_metrics = bg_metric_calculator.get_metrics()
        bg_RMSEs.append(bg_metrics['mean_RMSE'])
        bg_Rel_Errs.append(bg_metrics['mean_Rel_Err'])

    folds_bg_RMSEs.append(bg_RMSEs)
    folds_bg_Rel_Errs.append(bg_Rel_Errs)
    if var_args['synthetic_or_experimental'] == 'experimental':
        folds_inclusion_RMSEs.append(inclusion_RMSEs)
        folds_inclusion_Rel_Errs.append(inclusion_Rel_Errs)

    return bg_RMSEs, bg_Rel_Errs, inclusion_RMSEs, inclusion_Rel_Errs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(False)
    logging.info(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    logging.info(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    
    parser = argparse.ArgumentParser(description='PHEMA EDM2 Sweep')
    parser.add_argument('--save_dir', type=str, required=False, default=False, help='Path to config file and pickles')
    parser.add_argument('--folds_dir', type=str, required=False, default='pretrained_edm2_models', help='Path to directory containing pretrained model folders for cross-validation sweep')
    parser.add_argument('--val_batch_size', type=int, required=False, default=32, help='Validation batch size, overrides saved args if provided')
    parser.add_argument('--load_phema_metrics_json', type=str, required=False, default=False, help='Path to PHEMA metrics file to load for plotting only, in case already computed')
    parser.add_argument('--experimental_root_dir', type=str, required=False, default='/home/billy/Datasets/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms/', help='path to the root directory of the experimental dataset.')
    parser.add_argument('--synthetic_root_dir', type=str, required=False, default='/home/billy/Datasets/20250716_ImageNet_MSOT_Dataset/', help='path to the root directory of the synthetic dataset')
    args = parser.parse_args()

    if not args.load_phema_metrics_json:
        assert args.save_dir or args.folds_dir, "Either --save_dir or --folds_dir must be provided"

        if args.save_dir:
            save_dirs = [args.save_dir]
        else:
            # perform sweep over all pretrained models in directory
            save_dirs = [os.path.join(args.folds_dir, name) for name in os.listdir(args.folds_dir) 
                        if os.path.isdir(os.path.join(args.folds_dir, name))]
        
        folds_bg_RMSEs, folds_bg_Rel_Errs, folds_inclusion_RMSEs, folds_inclusion_Rel_Errs = [], [], [], []
        for i, save_dir in enumerate(save_dirs): # sweep over all pretrained models
            logging.info(f'Processing fold {i+1}/{len(save_dirs)}: {save_dir}')

            with open(os.path.join(save_dir, 'args.json'), 'r') as f:
                var_args = json.load(f)

            # override some args for phema sweep
            var_args['wandb_log'] = False
            var_args['save_test_examples'] = False
            var_args['save_dir'] = False
            if args.experimental_root_dir:
                var_args['experimental_root_dir'] = args.experimental_root_dir
            if args.synthetic_root_dir:
                var_args['synthetic_root_dir'] = args.synthetic_root_dir
            if args.val_batch_size:
                var_args['val_batch_size'] = args.val_batch_size

            if var_args['seed']:
                seed = var_args['seed']
            else:
                seed = np.random.randint(0, 2**32 - 1)
                var_args['seed'] = seed
            
            args_namespace = SimpleNamespace(**var_args)

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
                args_namespace, model_name=var_args['model'], 
                stats_path=os.path.join(var_args['experimental_root_dir'], 'dataset_stats.json')
            )
            (synthetic_datasets, synthetic_dataloaders, synthetic_transforms_dict) = uf.create_synthetic_dataloaders(
                args_namespace, model_name=var_args['model']
            )
            datasets = {'synthetic' : synthetic_datasets, 'experimental' : experimental_datasets}
            dataloaders = {'synthetic' : synthetic_dataloaders, 'experimental' : experimental_dataloaders}
            transforms_dict = {'synthetic' : synthetic_transforms_dict, 'experimental' : experimental_transforms_dict}

            match args_namespace.synthetic_or_experimental:
                case 'synthetic':
                    train_loader = dataloaders['synthetic']['train']
                case 'experimental':
                    train_loader = dataloaders['experimental']['train']

            phema_reconstruction_stds = (np.arange(20)+1).tolist()  # [1, 2, ..., 20]
            modules = reconstruct_edm2_phema_from_dir(
                save_dir, phema_reconstruction_stds, delete_pkls=False
            )
            
            bg_RMSEs, bg_Rel_Errs, inclusion_RMSEs, inclusion_Rel_Errs = evaluate_phema_sweep(
                modules, args_namespace, var_args, dataloaders, transforms_dict, device
            )

        results_dict_save_folder = args.save_dir if args.save_dir else args.folds_dir
        with open(os.path.join(results_dict_save_folder, 'phema_sweep_results.json'), 'w') as f:
            results_dict = {
                'phema_reconstruction_stds' : phema_reconstruction_stds,
                'bg_RMSEs' : folds_bg_RMSEs,
                'bg_Rel_Errs' : folds_bg_Rel_Errs,
                'inclusion_RMSEs' : folds_inclusion_RMSEs,
                'inclusion_Rel_Errs' : folds_inclusion_Rel_Errs,
            }
            json.dump(results_dict, f, indent=4)
        
    else:
        with open(args.load_phema_metrics_json, 'r') as f:
            results_dict = json.load(f)
        phema_reconstruction_stds = results_dict['phema_reconstruction_stds']
        folds_bg_RMSEs = results_dict['bg_RMSEs']
        folds_bg_Rel_Errs = results_dict['bg_Rel_Errs']
        folds_inclusion_RMSEs = results_dict['inclusion_RMSEs']
        folds_inclusion_Rel_Errs = results_dict['inclusion_Rel_Errs']
    
    # axis 0: folds, axis 1: phema stds, shape: (num_folds, num_stds)
    folds_bg_Rel_Errs = np.asarray(folds_bg_Rel_Errs)
    folds_bg_RMSEs = np.asarray(folds_bg_RMSEs)
    if len(folds_inclusion_RMSEs) > 0:
        folds_inclusion_Rel_Errs = np.asarray(folds_inclusion_Rel_Errs)
        folds_inclusion_RMSEs = np.asarray(folds_inclusion_RMSEs)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False, layout='constrained')

    ax[0].plot(phema_reconstruction_stds, folds_bg_RMSEs.mean(axis=0), marker='o', color='b')
    ax[0].fill_between(
        phema_reconstruction_stds,
        folds_bg_RMSEs.min(axis=0),
        folds_bg_RMSEs.max(axis=0),
        color='b', alpha=0.2
    )
    ax[0].set_xlabel('EMA (%)')
    ax[0].set_ylabel(r'RMSE (cm$^{-1}$)')
    ax[0].grid(True)
    ax[0].set_axisbelow(True)
    ax[0].set_xlim([min(phema_reconstruction_stds)-1, max(phema_reconstruction_stds)+1])

    ax[1].plot(phema_reconstruction_stds, folds_bg_Rel_Errs.mean(axis=0), marker='o', color='b', label='Background')
    ax[1].fill_between(
        phema_reconstruction_stds,
        folds_bg_Rel_Errs.min(axis=0),
        folds_bg_Rel_Errs.max(axis=0),
        color='b', alpha=0.2
    )
    ax[1].set_xlabel('EMA (%)')
    ax[1].set_ylabel(r'Relative Error (%)')
    ax[1].grid(True)
    ax[1].set_axisbelow(True)
    ax[1].set_xlim([min(phema_reconstruction_stds)-1, max(phema_reconstruction_stds)+1])

    if len(folds_inclusion_RMSEs) > 0:
        ax[0].plot(phema_reconstruction_stds, folds_inclusion_RMSEs.mean(axis=0), marker='o', color='r')
        ax[0].fill_between(
            phema_reconstruction_stds,
            folds_inclusion_RMSEs.min(axis=0),
            folds_inclusion_RMSEs.max(axis=0),
            color='r', alpha=0.2
        )
        ax[1].plot(phema_reconstruction_stds, folds_inclusion_Rel_Errs.mean(axis=0), marker='o', color='r', label='Inclusions')
        ax[1].fill_between(
            phema_reconstruction_stds,
            folds_inclusion_Rel_Errs.min(axis=0),
            folds_inclusion_Rel_Errs.max(axis=0),
            color='r', alpha=0.2
        )
        ax[1].legend()

    results_dict_save_folder = args.save_dir if args.save_dir else args.folds_dir
    plt.savefig(os.path.join(results_dict_save_folder, 'phema_sweep_results.pdf'), format='pdf')