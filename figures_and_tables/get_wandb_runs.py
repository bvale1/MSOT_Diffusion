
import os
import json
import itertools
import wandb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from copy import deepcopy

EXPERIMENTS = [
    "ImageNet_pretrain",
    "Digimouse_test",
    "Digimouse_extrusion_test",
    "experimental_from_scratch",
    "experimental_fine_tune",
]

MODELS = [
    'UNet_e2eQPAT',
    'EDM2',
    'UNet_diffusion_ablation',
]

FOLDS = [0, 1, 2, 3, 4]

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


# Load wandb credentials from project-root .env if present
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path, override=True)
wandb_key = os.getenv('WANDB_API_KEY')
if wandb_key:
    wandb.login(key=wandb_key)

api = wandb.Api(timeout=6000)

# Initialise empty result structure
metrics_dict: dict = {}
loss_curves_dict: dict = {}
for experiment in EXPERIMENTS:
    metrics_dict[experiment] = {}
    if experiment in ['experimental_from_scratch', 'experimental_fine_tune']:
        loss_curves_dict[experiment] = {}
    for model in MODELS:
        metrics_dict[experiment][model] = {}
        for metric in METRICS:
            if experiment in ['experimental_from_scratch', 'experimental_fine_tune']:
                metrics_dict[experiment][model][metric] = {
                    'bg': [],
                    'inclusion': [],
                }
                loss_curves_dict[experiment][model] = {'train_loss': [], 'val_loss': []}
            else:
                metrics_dict[experiment][model][metric] = {
                    'bg': [],  # will store fold as index and value as metric value, to be averaged later
                }


# Cache runs per project to avoid repeated API calls
project_runs: dict[str, list] = {}
# Optional: Filter by tags, state, etc. (see W&B API docs)
FILTER = {"state": "finished"}  # You can use {"state": "finished"} to get only completed runs
runs = api.runs("aisurrey_photoacoustics/MSOT_Diffusion2", filters=FILTER)



for run in runs:
    notes = run.notes
    name = run.name
    fold = int(name.split('fold')[-1])  # Assumes name format like "model_foldX"
    model = '_'.join(name.split('_')[:-1])  # Extract model name from run name
    # Assumes notes format like "experiment_model_foldX", find the string before the model name
    if f"_{model}_" in notes:
        experiment = notes.split(f"_{model}_")[0]
    else:
        raise ValueError(f"Unexpected notes format: {notes}")
    
    if fold not in FOLDS:
        print(f"Skipping fold {notes} not in FOLDS list")
        continue
    if model not in MODELS:
        print(f"Skipping model {notes} not in MODELS list")
        continue
    if experiment not in EXPERIMENTS:
        print(f"Skipping experiment {notes} not in EXPERIMENTS list")
        continue
    if run.state != "finished":
        print(f"Skipping model {notes} not finished (state={run.state})")
        continue

    # get test metrics logged to run.summary as '{mask}_{logging_prefix}_mean_{metric}'
    synthetic_or_experimental = 'experimental' if experiment in ['experimental_from_scratch', 'experimental_fine_tune'] else 'synthetic'
    logging_prefix = f'{synthetic_or_experimental}_test'
    mask_types = MASK_TYPES if synthetic_or_experimental == 'experimental' else ['bg']
    for metric in METRICS:
        for mask in mask_types:
            wandb_key = f'{mask}_{logging_prefix}_mean_{metric}'
            metrics_dict[experiment][model][metric][mask].append(run.summary.get(wandb_key))

    # get loss curve
    if synthetic_or_experimental == 'experimental':
        train_loss = run.history(
            x_axis='_step', keys=['mean_train_loss'], stream='default', pandas=True,
        )
        
        loss_curves_dict[experiment][model]['train_loss'].append(train_loss['mean_train_loss'].tolist())
        val_loss = run.history(
            x_axis='_step', keys=['mean_experimental_val_loss'], stream='default', pandas=True,
        )
        loss_curves_dict[experiment][model]['val_loss'].append(val_loss['mean_experimental_val_loss'].tolist())
        

# save results to json
with open('wandb_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)
with open('wandb_loss_curves.json', 'w') as f:
    json.dump(loss_curves_dict, f, indent=4)