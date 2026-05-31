import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_id', type=str, default='', action='store')
parser.add_argument('--lr', type=str, default='1e-3', help='learning rate')
parser.add_argument('--seed', type=str, default='42', help='seed for reproducibility')
parser.add_argument('--save_dir', type=str, default='20250723_diffusion', action='store')
parser.add_argument('--synthetic_or_experimental', type=str, choices=['experimental', 'synthetic', 'both'], default='synthetic', help='whether to use synthetic or experimental data')
parser.add_argument('--experimental_root_dir', type=str, default='/mnt/e/Dataset_for_Moving_beyond_simulation_data_driven_quantitative_photoacoustic_imaging_using_tissue_mimicking_phantoms/', help='path to the root directory of the experimental dataset.')
parser.add_argument('--synthetic_root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/20250327_ImageNet_MSOT_Dataset/', help='path to the root directory of the synthetic dataset')
parser.add_argument('--epochs', type=str, default='1000', help='number of training epochs, set to zero or one to test code')
parser.add_argument('--train_batch_size', type=str, default='16', help='batch size for training')
parser.add_argument('--val_batch_size', type=str, default='64', help='batch size for inference, 4x train_batch_size should have similar device memory requirements')
parser.add_argument('--warmup_period', type=str, default='1', help='warmup period for the learning rate, must be int greater than 0')
parser.add_argument('--model', type=str, choices=['UNet_e2eQPAT', 'UNet_diffusion_ablation', 'EDM2'], default='UNet_e2eQPAT', help='model to train')
parser.add_argument('--data_normalisation', choices=['standard', 'minmax'], default='standard', help='normalisation method for the data')
parser.add_argument('--fold', type=str, choices=['0', '1', '2', '3', '4'], default='0', help='fold for cross-validation, only used for experimental data')
parser.add_argument('--wandb_notes', type=str, default='None', help='optional, comment for wandb')
parser.add_argument('--load_checkpoint_dir', type=str, default=None, help='path to load a model checkpoint')
parser.add_argument('--boft_rank', type=str, default='0', help='rank for butterfly orthogonal fine tuning layers, 0 means no BOFT')
parser.add_argument('--git_hash', type=str, default=None, help='specifiy an older version of the codebase')
parser.add_argument('--l2_regularisation', type=str, default='0.0', help='weight for L2 regularisation loss term')
parser.add_argument('--wl_conditioning', default=False, help='use wavelength conditioning in diffusion models', action='store_true')
parser.add_argument('--skip_test', default=False, action='store_true', help='skip testing at the end of training')
parser.add_argument('--resume_training_from', type=str, default=None, help='path to a previous save_dir to resume training from')
args = parser.parse_args()
# ARGUMENTS
save_dir = args.save_dir + args.cluster_id

lr = '1e-3' if args.synthetic_or_experimental == 'synthetic' else '1e-4'
image_size = '256' if args.synthetic_or_experimental == 'synthetic' else '288'
#data_normalisation = 'minmax' if args.model == 'DDIM' else 'standard'
std_data = '0.5' if args.model == 'EDM2' else '1.0'
phema_reconstruction_std = '0.07'

# CREATE FILE TO SAVE SIMULATION
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
os.chdir(save_dir)

# CLONE LATEST FROM REPO
#subprocess.run(['rm', '-rf', 'MSOT_diffusion'])
subprocess.run(
    ['git', 'clone', '--recurse-submodules', '-b', 'main', 'git@github.com:bvale1/MSOT_diffusion.git']
)
if args.git_hash is not None:
    subprocess.run(['git', '-C', 'MSOT_diffusion', 'checkout', args.git_hash])
    subprocess.run(['git', '-C', 'MSOT_diffusion', 'submodule', 'update', '--init', '--recursive'])

subprocess.run(
    ['pip', 'install', 'MSOT_diffusion/denoising-diffusion-pytorch']
)

# GET HASH OF THE CLONED COMMIT
git_hash = subprocess.check_output(
    ['git', '-C', 'MSOT_diffusion', 'rev-parse', 'HEAD']
).decode('utf-8').strip()
print('git hash: ' + git_hash)

cmd = [
    'python3', 'MSOT_diffusion/run_model.py',
    '--synthetic_or_experimental', args.synthetic_or_experimental,
    '--experimental_root_dir', args.experimental_root_dir,
    '--synthetic_root_dir', args.synthetic_root_dir,
    '--git_hash', git_hash,
    '--epochs', args.epochs,
    '--train_batch_size', args.train_batch_size,
    '--val_batch_size', args.val_batch_size,
    '--image_size', image_size,
    '--save_test_examples',
    '--wandb_log',
    '--lr',  args.lr,
    '--seed', args.seed,
    '--save_dir', save_dir,
    '--warmup_period', args.warmup_period,
    '--model', args.model,
    '--data_normalisation', args.data_normalisation,
    '--fold', args.fold,
    '--wandb_notes', args.wandb_notes,
    '--predict_fluence',
    '--no_lr_scheduler',
    #'--freeze_encoder',
    #'--attention',
    '--std_data', std_data,
    '--phema_reconstruction_std', phema_reconstruction_std,
    '--boft_rank', args.boft_rank,
    '--l2_regularisation', args.l2_regularisation
]

if args.load_checkpoint_dir:
    cmd.extend(['--load_checkpoint_dir', args.load_checkpoint_dir])
if args.wl_conditioning:
    cmd.extend(['--wl_conditioning'])
if args.resume_training_from:
    cmd.extend(['--resume_training_from', args.resume_training_from])
if args.skip_test:
    cmd.extend(['--skip_test'])

subprocess.run(cmd)