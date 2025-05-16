import torch
import json
import os
import argparse
import sys

import denoising_diffusion_pytorch as ddp

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
import utility_functions as uf
import end_to_end_phantom_QPAT.utils.networks as e2eQPAT_networks


def compute_outputs(models_dirs, sample_indices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        raise ValueError('cuda is not available')

    outputs_dict = {model: {} for model, _ in models_dirs}

    for model_name, dir in models_dirs:
        # ==================== args ====================
        with open(os.path.join(dir.split('/')[0], 'args.json'), 'r') as f:
            args = json.load(f)
        args['wandb_log'] = False
        args['synthetic_root_dir'] = '/home/wv00017/MSOT_Diffusion/20250327_ImageNet_MSOT_Dataset/'
        args['experimental_root_dir'] = '/mnt/e/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms/'
        args = argparse.Namespace(**args)
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
        # ===================== model ====================
        image_size = (args.image_size, args.image_size)
        channels = datasets['synthetic']['test'][0][0].shape[-3]
        out_channels = channels * 2 if args.predict_fluence else channels
        match args.model:
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
            case 'DDIM':
                out_channels = channels * 2 if args.predict_fluence else channels
                model = ddp.Unet(
                    dim=32, channels=out_channels, out_dim=out_channels,
                    self_condition=args.self_condition, image_condition=True, 
                    image_condition_channels=channels, full_attn=False, flash_attn=False
                )
                diffusion = ddp.GaussianDiffusion(
                    # objecive='pred_v' predicts the velocity field, objective='pred_noise' predicts the noise
                    model, image_size=image_size, timesteps=1000,
                    sampling_timesteps=100, objective=args.objective, auto_normalize=False,
                )
        
        model.load_state_dict(torch.load(dir, weights_only=True))
        model.eval()
        model.to(device)
        
        X, mu_a, fluence, wavelength_nm, mask = [], [], [], [], []
        for sample_index in sample_indices:
            X_, mu_a_, fluence_, wavelength_nm_, mask_ = datasets[args.synthetic_or_experimental]['test'][sample_index][:5]
            X.append(X_)
            mu_a.append(mu_a_)
            fluence.append(fluence_)
            wavelength_nm.append(wavelength_nm_)
            mask.append(mask_)
        
        X = torch.stack(X, dim=0).to(device)
        mu_a = torch.stack(mu_a, dim=0)
        mask = torch.stack(mask, dim=0)
        fluence = torch.stack(fluence, dim=0)
        wavelength_nm = torch.stack(wavelength_nm, dim=0).to(device)
        
        with torch.no_grad():
            match args.model:
                case 'UNet_e2eQPAT':
                    Y_hat = model(X)
                case 'UNet_wl_pos_emb':
                    Y_hat = model(X, wavelength_nm.squeeze())
                case 'UNet_diffusion_ablation':
                    Y_hat = model(X, torch.zeros(wavelength_nm.shape[0], device=device))
                case 'DDIM':
                    Y_hat = diffusion.sample(batch_size=X.shape[0], x_cond=X)
        
        mu_a_hat = Y_hat[:, 0:1]
        outputs_dict[model_name]['X'] = transforms_dict[args.synthetic_or_experimental]['normalise_x'].inverse(X.cpu()).squeeze().numpy()
        outputs_dict[model_name]['mu_a'] = transforms_dict[args.synthetic_or_experimental]['normalise_mu_a'].inverse(mu_a.cpu()).squeeze().numpy()
        outputs_dict[model_name]['mu_a_hat'] = transforms_dict[args.synthetic_or_experimental]['normalise_mu_a'].inverse(mu_a_hat.cpu()).squeeze().numpy()
        outputs_dict[model_name]['fluence'] = transforms_dict[args.synthetic_or_experimental]['normalise_fluence'].inverse(fluence.cpu()).squeeze().numpy()
        outputs_dict[model_name]['wavelength_nm'] = wavelength_nm.cpu().squeeze().numpy()
        outputs_dict[model_name]['mask'] = mask.cpu().squeeze().numpy()
        if args.predict_fluence:
            fluence_hat = Y_hat[:, 1:2]
            outputs_dict[model_name]['fluence_hat'] = transforms_dict[args.synthetic_or_experimental]['normalise_fluence'].inverse(fluence_hat.cpu()).squeeze().numpy()
    return outputs_dict, datasets[args.synthetic_or_experimental]['test'].cfg['dx'] * 1e3 # [m] -> [mm]