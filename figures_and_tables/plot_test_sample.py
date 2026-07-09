import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from edm2.training.networks_edm2 import Precond
from edm2.training.networks_edm2 import UNet as EDM2_UNet
from edm2.generate_images import edm_sampler
import end_to_end_phantom_QPAT.utils.networks as e2eQPAT_networks
import utility_functions as uf
from experiments.weights_paths import checkpoint_dirs

MODELS = [
    'EDM2',
    'UNet_e2eQPAT',
    'UNet_diffusion_ablation',
]
models = [
    'UNet_e2eQPAT',
    'EDM2',
    'UNet_diffusion_ablation',
]

EXPERIMENTS = [
    "ImageNet_pretrain",
    "Digimouse_test",
    "Digimouse_extrusion_test",
    "experimental_from_scratch",
    "experimental_fine_tune",
]
experiments = [
    "ImageNet_pretrain",
]
# experiments = [
#     "experimental_from_scratch",
#     "experimental_fine_tune",
# ]

FOLDS = [0, 1, 2, 3, 4]
fold = 0

# local dataset root directories
DATASETS = {
    "ImageNet": "/home/billy/Projects/MSOT_Diffusion/20260522_ImageNet_MSOT_Dataset",
    "Digimouse": "/home/billy/Projects/MSOT_Diffusion/20260522_digimouse_MSOT_Dataset",
    "Digimouse_extrusion": "/home/billy/Projects/MSOT_Diffusion/20260522_digimouse_extrusion_MSOT_Dataset",
    "e2eQPAT": "/home/billy/Datasets/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms",
}

# ImageNet synthetic
sample_name = '__mnt__fast__datasets__still__ImageNet__ILSVRC2012__TrainingSet__n02101006__n02101006_993.JPEG'
# exprimental
#sample_name = "P.2.1_700"


font_size = 15 # controls the text scaling of the figure
base_dir = os.path.dirname(os.path.abspath(__file__))
savename = os.path.join(
    base_dir, f"test_sample_{sample_name}_{'_'.join(models)}_{'_'.join(experiments)}_fold{fold}.pdf"
)

combos = list(product(models, experiments))
synthetic_experiments = ["ImageNet_pretrain", "Digimouse_test", "Digimouse_extrusion_test"]
if sorted(models) != sorted(MODELS) or not (
        (len(experiments) == 1 and experiments[0] in synthetic_experiments)
        or experiments == ["experimental_from_scratch", "experimental_fine_tune"]
    ):
    raise ValueError('models must contain all three models, and experiments must \
        be a single synthetic experiment or both experimental experiments')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

X_plot = None; mu_a_plot = None; dx = None
outputs = {}
for model_name, experiment in combos:
    checkpoint_dir = os.path.join(base_dir, checkpoint_dirs[model_name][experiment][fold])
    with open(os.path.join(checkpoint_dir, 'args.json'), 'r') as f:
        args = argparse.Namespace(**json.load(f))
    # point dataset paths at local copies and disable logging/saving
    args.wandb_log = False
    args.save_test_examples = False
    args.fold = str(fold)
    args.experimental_root_dir = DATASETS['e2eQPAT']
    synthetic_dataset = 'Digimouse' if experiment in ['Digimouse_test'] else ('Digimouse_extrusion' if experiment in ['Digimouse_extrusion_test'] else 'ImageNet')
    args.synthetic_root_dir = DATASETS[synthetic_dataset]

    # 1) initialize the test datasets and dataloaders
    if args.synthetic_or_experimental == 'experimental':
        (datasets, dataloaders, transforms_dict) = uf.create_e2eQPAT_dataloaders(
            args, model_name=args.model,
            stats_path=os.path.join(args.experimental_root_dir, 'dataset_stats.json')
        )
    else:
        (datasets, dataloaders, transforms_dict) = uf.create_synthetic_dataloaders(
            args, model_name=args.model
        )
    examples_dataset = datasets['test']
    
    if args.synthetic_or_experimental == 'experimental':
        sample_names = ['.'.join(file.split('/')[-1].split('.')[:-1]) for file in examples_dataset.files]
    else:
        sample_names = [sample.decode('utf-8') for sample in examples_dataset.samples]
    
    idx = sample_names.index(sample_name)
    (X, mu_a, fluence, wavelength_nm, mask, _, file) = examples_dataset[idx][:7]
    X = X.unsqueeze(0).to(device)
    wavelength_nm = wavelength_nm.unsqueeze(0).to(device)

    # 2) initialize the models and load the weights
    channels = X.shape[-3]
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
    model.load_state_dict(
        uf.load_best_checkpoint_from(checkpoint_dir), strict=False
    )
    model.to(device)
    model.eval()

    # 3) compute outputs for a sample of test data (from synthetic or experimental) using the trained models
    # dedicated generator so EDM2 always samples the same initial noise,
    # using the same seed and convention as compute_outputs.py
    noise_rng = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
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
                # replay the batched inference of compute_outputs.py
                # (val_batch_size=8) so this sample's output is identical to the
                # one its metrics in test_sample_metrics.csv were computed on
                batch_size = 8
                for b in range(idx // batch_size + 1):
                    n = min(batch_size, len(examples_dataset) - b * batch_size)
                    noise = torch.randn(
                        (n, out_channels, args.image_size, args.image_size),
                        device=device, generator=noise_rng
                    )
                batch_start = (idx // batch_size) * batch_size
                X_batch = torch.stack(
                    [examples_dataset[i][0] for i in range(batch_start, batch_start + n)]
                ).to(device)
                wavelength_nm_batch = torch.stack(
                    [examples_dataset[i][3] for i in range(batch_start, batch_start + n)]
                ).to(device)
                wavelength_nm_onehot = torch.zeros(
                    (wavelength_nm_batch.shape[0], 1000), dtype=torch.float32, device=device
                )
                wavelength_nm_onehot[:, wavelength_nm_batch.squeeze()] = 1.0
                Y_hat = edm_sampler(
                    model, noise, x_cond=X_batch, labels=wavelength_nm_onehot, num_steps=16
                )
                Y_hat = Y_hat[idx - batch_start].unsqueeze(0)
    mu_a_hat = Y_hat[:, 0:1].detach().cpu()

    mu_a_hat = transforms_dict['normalise_mu_a'].inverse(mu_a_hat).squeeze().numpy()
    mu_a_hat += 1e-2 # convert mu_a from m^-1 to cm^-1
    outputs[(model_name, experiment)] = mu_a_hat
    if X_plot is None:
        X_plot = transforms_dict['normalise_x'].inverse(X.detach().cpu()).squeeze().numpy()
        mu_a_plot = transforms_dict['normalise_mu_a'].inverse(mu_a).squeeze().numpy()
        mu_a_plot += 1e-2 # convert mu_a from m^-1 to cm^-1
        dx = examples_dataset.cfg['dx'] * 1e3 # [m] -> [mm]

    del model
    torch.cuda.empty_cache()

# 4) plot the outputs in a mosiac like this:

# if plotting ImageNet_pretrain, Digimouse_test, or Digimouse_extrusion_test, the mosaic should look like this:
#  | input                     |     ground truth           |   line profile comparison  |
#  | UNet_e2eQPAT              |           EDM2             |    UNet_diffusion_ablation |

# if plotting experimental_from_scratch and experimental_fine_tune, the mosaic should look like this:
#  | input                     |      ground truth          |   line profile comparison  |
#                                       From scratch
#  | UNet_e2eQPAT              |           EDM2             |    UNet_diffusion_ablation |
#                                       fine-tuned
#  | UNet_e2eQPAT              |           EDM2             |    UNet_diffusion_ablation |

# all other combinations of models and experiments are invalid and will raise a ValueError
# the position of all line profiles are shown on the images in the same color and linestyle as in the line profile comparison plot

v_min_Y = min(np.min(mu_a_plot), min(np.min(out) for out in outputs.values()))
v_max_Y = max(np.max(mu_a_plot), max(np.max(out) for out in outputs.values()))
extent = [-dx*X_plot.shape[-2]/2, dx*X_plot.shape[-2]/2,
          -dx*X_plot.shape[-1]/2, dx*X_plot.shape[-1]/2]
# all line profiles are taken along the middle row of the images
profile_row = mu_a_plot.shape[0]//2
profile_z = extent[2] + (profile_row + 0.5)*dx

colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']
combo_colors = {combo : colors[i] for i, combo in enumerate(combos)}
if len(experiments) == 1:
    combo_labels = {combo : combo[0].replace('_', ' ') for combo in combos}
else:
    experiment_display = {
        'experimental_from_scratch': 'from scratch',
        'experimental_fine_tune': 'fine-tuned',
    }
    combo_labels = {
        combo : f'{combo[0]} {experiment_display[combo[1]]}'.replace('_', ' ') for combo in combos
    }

plt.rcParams.update({'font.size': font_size})
# a single grid so the columns of every row align exactly
n_rows = len(experiments) + 1
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows), layout='constrained')
input_ax = axes[0, 0]
gt_ax = axes[0, 1]
profile_ax = axes[0, 2]
output_axes = {
    (model, experiment) : axes[i+1, j]
    for i, experiment in enumerate(experiments) for j, model in enumerate(models)
}
# constrained layout does not account for the appended colorbar axes or the
# row headings, so pad the panels to make room
fig.get_layout_engine().set(wspace=0.15, hspace=0.1)
if len(experiments) == 2:
    # row headings centred above the middle image of each experiment row
    for i, heading in enumerate(['From scratch', 'Fine-tuned']):
        axes[i+1, 1].text(
            0.5, 1.12, heading, transform=axes[i+1, 1].transAxes,
            ha='center', va='bottom', fontsize=font_size*1.8
        )

img = input_ax.imshow(
    X_plot, cmap='binary_r', origin='lower', extent=extent
)
input_ax.set_title(r'Reconstruction $p_{0}^{\mathrm{rec}}$')
# colorbar the same height as the image, with a horizontal label above it
divider = make_axes_locatable(input_ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(img, cax=cax)
cbar.ax.set_title(r'Pa J$^{-1}$')
input_ax.set_ylabel('z (mm)')

img = gt_ax.imshow(
    mu_a_plot, cmap='binary_r', vmin=v_min_Y, vmax=v_max_Y,
    origin='lower', extent=extent
)
gt_ax.set_title(r'Reference $\mu_{\mathrm{a}}$')
divider = make_axes_locatable(gt_ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(img, cax=cax)
cbar.ax.set_title(r'cm$^{-1}$')

for combo in combos:
    ax = output_axes[combo]
    img = ax.imshow(
        outputs[combo], cmap='binary_r', vmin=v_min_Y, vmax=v_max_Y,
        origin='lower', extent=extent
    )
    ax.set_title(combo[0].replace('_', ' '))
    ax.axhline(profile_z, color=combo_colors[combo], linestyle='dashed')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.set_title(r'cm$^{-1}$')

# hide tics shared with the image to the left or the row below
input_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
gt_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
gt_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
profile_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
for i, experiment in enumerate(experiments):
    for j, model in enumerate(models):
        ax = axes[i+1, j]
        if j == 0:
            ax.set_ylabel('z (mm)')
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if i+1 == len(experiments):
            ax.set_xlabel('x (mm)')
        else:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

line_profile_axis = np.arange(-dx*mu_a_plot.shape[-1]/2, dx*mu_a_plot.shape[-1]/2, dx)
line_profiles = [mu_a_plot[profile_row, :]]
line_profiles += [outputs[combo][profile_row, :] for combo in combos]
profile_ax.plot(
    line_profile_axis, line_profiles[0],
    label=r'Reference $\mu_{\mathrm{a}}$', color='black', linestyle='solid'
)
for i, combo in enumerate(combos):
    profile_ax.plot(
        line_profile_axis, line_profiles[i+1],
        # each image shows its own profile line, so the legend entries are
        # redundant; uncomment to label every curve
        #label=combo_labels[combo],
        color=combo_colors[combo], linestyle='dashed'
    )
profile_ax.set_title('Line profile')
profile_ax.set_box_aspect(1)
profile_ax.set_ylabel(r'cm$^{-1}$')
profile_ax.grid(True)
profile_ax.set_axisbelow(True)
profile_ax.set_xlim(extent[0], extent[1])
# extend the y axis above the data so the legend never overlaps the profiles,
# scaling the headroom with the number of legend entries
n_legend_entries = len(profile_ax.get_legend_handles_labels()[1])
profile_min = min(np.min(profile) for profile in line_profiles)
profile_max = max(np.max(profile) for profile in line_profiles)
profile_ax.set_ylim(top=profile_max + 0.15*n_legend_entries*(profile_max - profile_min))
profile_ax.legend(loc='upper center')

# the profile plot's y labels widen the gap before the third column, so freeze
# the solved layout and shift the third image of each output row left until the
# spacing between the images is even
fig.canvas.draw()
fig.set_layout_engine('none')
for row in axes[1:]:
    x0 = [ax.get_position().x0 for ax in row]
    shift = (x0[2] - x0[1]) - (x0[1] - x0[0])
    pos = row[2].get_position(original=True)
    row[2].set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])

fig.savefig(savename, dpi=300, bbox_inches='tight', format='pdf')
