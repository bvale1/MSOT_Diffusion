import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from itertools import product

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from edm2.training.networks_edm2 import Precond
from edm2.training.networks_edm2 import UNet as EDM2_UNet
from edm2.generate_images import edm_sampler
import end_to_end_phantom_QPAT.utils.networks as e2eQPAT_networks
import utility_classes as uc
import utility_functions as uf
from epoch_steps import reconstruct_edm2_phema_from_dir
from experiments.weights_paths import (
    original_weights_e2eQPAT, from_scratch_e2eQPAT, checkpoint_dirs
)

# models = [
#     'UNet_e2eQPAT',
#     'EDM2',
#     'UNet_diffusion_ablation',
# ]
models = ['UNet_e2eQPAT']

# experiments = [
#     "ImageNet_pretrain",
#     "Digimouse_test",
#     "Digimouse_extrusion_test",
#     "experimental_from_scratch",
#     "experimental_fine_tune",
# ]
experiments = [
    "experimental_fine_tune",
    "experimental_from_scratch",
]

folds = [0, 1, 2, 3, 4]

METRICS = [
    'RMSE',
    'MAE',
    'Rel_Err',
    'PSNR',
]

MASK_TYPES = [
    'bg',
    'inclusion',
]
# MASK_TYPES = ['bg']

# how to aggregate the per-sample metrics over the test set into one row per fold
# for wandb_metrics.csv: 'mean' or 'median'
#AGG = 'mean'
AGG = 'median'

# also evaluate the reference U-Nets from the original e2eQPAT study (Groehl et
# al.), scored on the experimental test set and written as a separate model
INCLUDE_ORIGINAL_E2EQPAT = True
ORIGINAL_MODEL_NAME = 'UNet_e2eQPAT_original'
ORIGINAL_EXPERIMENT = 'experimental_from_scratch'

# local dataset root directories
DATASETS = {
    "ImageNet": "/home/billy/Projects/MSOT_Diffusion/20260522_ImageNet_MSOT_Dataset",
    "Digimouse": "/home/billy/Projects/MSOT_Diffusion/20260522_digimouse_MSOT_Dataset",
    "Digimouse_extrusion": "/home/billy/Projects/MSOT_Diffusion/20260522_digimouse_extrusion_MSOT_Dataset",
    "e2eQPAT": "/home/billy/Datasets/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms",
}

# which dataset each experiment is tested on
EXPERIMENT_DATASETS = {
    "ImageNet_pretrain": "ImageNet",
    "Digimouse_test": "Digimouse",
    "Digimouse_extrusion_test": "Digimouse_extrusion",
    "experimental_from_scratch": "e2eQPAT",
    "experimental_fine_tune": "e2eQPAT",
}

base_dir = os.path.dirname(os.path.abspath(__file__))
savename = os.path.join(base_dir, 'test_sample_metrics.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise ValueError('cuda is not available')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

metric_rows = []
for model_name, experiment, fold in product(models, experiments, folds):
    print(f'processing model: {model_name}, experiment: {experiment}, fold: {fold}')
    checkpoint_dir = os.path.join(base_dir, checkpoint_dirs[model_name][experiment][fold])
    with open(os.path.join(checkpoint_dir, 'args.json'), 'r') as f:
        args = argparse.Namespace(**json.load(f))
    # point dataset paths at local copies and disable logging/saving
    args.wandb_log = False
    args.save_test_examples = False
    args.fold = str(fold)
    args.val_batch_size = 8 # the training value OOMs on a 24 GB local GPU

    # 1) initialize the test dataset and dataloader
    dataset_name = EXPERIMENT_DATASETS[experiment]
    experimental = dataset_name == 'e2eQPAT'
    if experimental:
        args.experimental_root_dir = DATASETS[dataset_name]
        (datasets, dataloaders, transforms_dict) = uf.create_e2eQPAT_dataloaders(
            args, model_name=args.model,
            stats_path=os.path.join(args.experimental_root_dir, 'dataset_stats.json')
        )
    else:
        args.synthetic_root_dir = DATASETS[dataset_name]
        (datasets, dataloaders, transforms_dict) = uf.create_synthetic_dataloaders(
            args, model_name=args.model
        )
    test_dataloader = dataloaders['test']

    # 2) initialize the models and load the weights
    channels = datasets['test'][0][0].shape[-3]
    out_channels = channels * 2 if args.predict_fluence else channels
    match args.model:
        case 'UNet_e2eQPAT':
            model = e2eQPAT_networks.RegressionUNet(
                in_channels=channels,
                out_channels=out_channels,
                initial_filter_size=64,
                kernel_size=3
            )
        case 'UNet_diffusion_ablation':
            model = EDM2_UNet(
                img_resolution=256, # use 256 for backward compatibility with pretrained weights; doesn't affect architecture
                img_channels_in=channels,
                img_channels_out=out_channels,
                label_dim=1000 if args.wl_conditioning else 0,
                model_channels=64,
                attn_resolutions=[16, 8] if args.attention else [],
                noise_emb=False,
            )
        case 'EDM2':
            label_dim = 1000 if args.wl_conditioning else 0
            in_channels = out_channels+1 # plus 1 for conditional information
            model = Precond(
                img_resolution=256, # use 256 for backward compatibility with pretrained weights; doesn't affect architecture
                img_channels_in=in_channels,
                img_channels_out=out_channels,
                label_dim=label_dim,
                model_channels=64,
                attn_resolutions=[16, 8] if args.attention else [],
                use_fp16=False,
                sigma_data=0.5
            )
            if not args.attention:
                uf.remove_attention(model.unet)
    if model_name in ['EDM2', 'UNet_diffusion_ablation']:
        # the net tested in run_model.py is the phEMA reconstruction, so
        # rebuild it from the latest snapshots rather than model_state_dict
        latest_state = reconstruct_edm2_phema_from_dir(
            checkpoint_dir, [args.phema_reconstruction_std], delete_pkls=False
        )[0]['net'].state_dict()
    else:
        latest_state = torch.load(
            os.path.join(checkpoint_dir, 'latest_checkpoint.pt'),
            weights_only=True
        )['model_state_dict']
    model.load_state_dict(latest_state, strict=False)
    model.to(device)
    model.eval()

    # 3) compute outputs and per-sample metrics on the entire test set,
    #    the same way test_epoch does so they match the wandb summary metrics
    bg_metric_calculator = uc.TestMetricCalculator()
    inclusion_metric_calculator = uc.TestMetricCalculator()
    sample_names = []
    # dedicated generator so EDM2 always samples the same initial noise,
    # regardless of any other RNG consumers (e.g. dataloader workers)
    noise_rng = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        for batch in test_dataloader:
            X = batch[0].to(device); mu_a = batch[1].to(device)
            wavelength_nm = batch[3].to(device)
            files = batch[6]
            # sample name is the file basename without the extension (synthetic
            # sample names have no extension and are kept as they are)
            sample_names += [file.split('/')[-1].rsplit('.', 1)[0] for file in files]

            match args.model:
                case 'UNet_e2eQPAT':
                    Y_hat = model(X)
                case 'UNet_diffusion_ablation':
                    if args.wl_conditioning:
                        wavelength_nm_onehot = torch.zeros(
                            (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                        )
                        wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                        Y_hat = model(X, class_labels=wavelength_nm_onehot)
                    else:
                        Y_hat = model(X)
                case 'EDM2':
                    wavelength_nm_onehot = torch.zeros(
                        (wavelength_nm.shape[0], 1000), dtype=torch.float32, device=device
                    )
                    wavelength_nm_onehot[:, wavelength_nm.squeeze()] = 1.0
                    noise = torch.randn(
                        (X.shape[0], out_channels, args.image_size, args.image_size),
                        device=device, generator=noise_rng
                    )
                    Y_hat = edm_sampler(
                        model, noise, x_cond=X, labels=wavelength_nm_onehot, num_steps=16
                    )

            mu_a_hat = Y_hat[:, 0:1]
            bg_metric_calculator(
                Y=mu_a, Y_hat=mu_a_hat, Y_transform=transforms_dict['normalise_mu_a'],
                Y_mask=batch[4] # background mask
            )
            if experimental: # synthetic datasets have no inclusion masks
                inclusion_metric_calculator(
                    Y=mu_a, Y_hat=mu_a_hat, Y_transform=transforms_dict['normalise_mu_a'],
                    Y_mask=batch[5] # inclusion mask
                )

    # one row per test sample, inclusion metrics are left as NaN for synthetic data
    for i, sample_name in enumerate(sample_names):
        row = {
            'sample_name': sample_name, 'experiment': experiment,
            'model': model_name, 'fold': fold,
            **{f'bg_{metric}': bg_metric_calculator.metrics[metric][i] for metric in METRICS},
        }
        if experimental:
            row.update({
                f'inclusion_{metric}': inclusion_metric_calculator.metrics[metric][i] for metric in METRICS
            })
        metric_rows.append(row)

    del model
    torch.cuda.empty_cache()

# evaluate the original e2eQPAT reference U-Nets on the experimental test set.
# these are raw RegressionUNet state_dicts (no args.json), so they are handled
# separately from the main loop. args are borrowed from the matching from-scratch
# e2eQPAT run so the dataloader is identical, with std_data=1.0 to reproduce the
# plain standardisation the original models were trained with
if INCLUDE_ORIGINAL_E2EQPAT and 'experimental_from_scratch' in experiments:
    for fold in folds:
        print(f'processing model: {ORIGINAL_MODEL_NAME}, experiment: {ORIGINAL_EXPERIMENT}, fold: {fold}')
        with open(os.path.join(base_dir, from_scratch_e2eQPAT[fold], 'args.json'), 'r') as f:
            args = argparse.Namespace(**json.load(f))
        args.wandb_log = False
        args.save_test_examples = False
        args.fold = str(fold)
        args.val_batch_size = 8
        args.experimental_root_dir = DATASETS['e2eQPAT']
        args.model = 'UNet_e2eQPAT'
        args.predict_fluence = True
        args.data_normalisation = 'standard'
        args.std_data = 1.0

        (datasets, dataloaders, transforms_dict) = uf.create_e2eQPAT_dataloaders(
            args, model_name=args.model,
            stats_path=os.path.join(args.experimental_root_dir, 'dataset_stats.json')
        )
        test_dataloader = dataloaders['test']

        channels = datasets['test'][0][0].shape[-3]
        out_channels = channels * 2 if args.predict_fluence else channels
        model = e2eQPAT_networks.RegressionUNet(
            in_channels=channels,
            out_channels=out_channels,
            initial_filter_size=64,
            kernel_size=3
        )
        model.load_state_dict(
            torch.load(original_weights_e2eQPAT[fold], map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        bg_metric_calculator = uc.TestMetricCalculator()
        inclusion_metric_calculator = uc.TestMetricCalculator()
        sample_names = []
        with torch.no_grad():
            for batch in test_dataloader:
                X = batch[0].to(device); mu_a = batch[1].to(device)
                files = batch[6]
                sample_names += [file.split('/')[-1].rsplit('.', 1)[0] for file in files]
                Y_hat = model(X)
                mu_a_hat = Y_hat[:, 0:1]
                bg_metric_calculator(
                    Y=mu_a, Y_hat=mu_a_hat, Y_transform=transforms_dict['normalise_mu_a'],
                    Y_mask=batch[4] # background mask
                )
                inclusion_metric_calculator(
                    Y=mu_a, Y_hat=mu_a_hat, Y_transform=transforms_dict['normalise_mu_a'],
                    Y_mask=batch[5] # inclusion mask
                )

        for i, sample_name in enumerate(sample_names):
            row = {
                'sample_name': sample_name, 'experiment': ORIGINAL_EXPERIMENT,
                'model': ORIGINAL_MODEL_NAME, 'fold': fold,
                **{f'bg_{metric}': bg_metric_calculator.metrics[metric][i] for metric in METRICS},
                **{f'inclusion_{metric}': inclusion_metric_calculator.metrics[metric][i] for metric in METRICS},
            }
            metric_rows.append(row)

        del model
        torch.cuda.empty_cache()

# 4) save one row per sample to csv, only overwriting the experiments computed here
metric_columns = ['sample_name', 'experiment', 'model', 'fold'] + [
    f'{mask}_{metric}' for mask in MASK_TYPES for metric in METRICS
]
key_columns = ['experiment', 'model', 'fold']
new_metrics_df = pd.DataFrame(metric_rows, columns=metric_columns)
metrics_df = new_metrics_df
if os.path.exists(savename):
    old_df = pd.read_csv(savename)
    merged = old_df.merge(
        new_metrics_df[key_columns].drop_duplicates(), on=key_columns, how='left', indicator=True
    )
    old_df = old_df[(merged['_merge'] == 'left_only').to_numpy()]
    metrics_df = pd.concat([old_df, new_metrics_df], ignore_index=True)
metrics_df = metrics_df.sort_values(['experiment', 'model', 'fold', 'sample_name'])
metrics_df.to_csv(savename, index=False)
print(f'saved {len(metrics_df)} rows to {savename}')

# 5) aggregate over the test samples to one row per fold (mean matches the
#    wandb '{mask}_{synthetic_or_experimental}_test_mean_{metric}' summary metrics)
#    and update wandb_metrics.csv, only overwriting the experiments computed here
value_columns = [f'{mask}_{metric}' for mask in MASK_TYPES for metric in METRICS]
fold_agg = new_metrics_df.groupby(key_columns, as_index=False)[value_columns].agg(AGG)

wandb_csv = os.path.join(base_dir, 'wandb_metrics.csv')
if os.path.exists(wandb_csv):
    wandb_df = pd.read_csv(wandb_csv)
    merged = wandb_df.merge(fold_agg[key_columns], on=key_columns, how='left', indicator=True)
    wandb_df = wandb_df[(merged['_merge'] == 'left_only').to_numpy()]
    fold_agg = pd.concat([wandb_df, fold_agg], ignore_index=True)
fold_agg = fold_agg.sort_values(key_columns)
fold_agg.to_csv(wandb_csv, index=False)
print(f'saved {len(fold_agg)} rows to {wandb_csv}')
