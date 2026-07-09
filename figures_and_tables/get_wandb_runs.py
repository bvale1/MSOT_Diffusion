
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

LOSS_CURVE_METRICS = [
    'mean_train_loss',
    'mean_experimental_val_loss',
    'inclusion_experimental_val_mean_RMSE',
    'inclusion_experimental_val_mean_MAE',
    'inclusion_experimental_val_mean_Rel_Err',
    'inclusion_experimental_val_mean_PSNR',
    'bg_experimental_val_mean_RMSE',
    'bg_experimental_val_mean_MAE',
    'bg_experimental_val_mean_Rel_Err',
    'bg_experimental_val_mean_PSNR',
]

# Load wandb credentials from project-root .env if present
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path, override=True)
wandb_key = os.getenv('WANDB_API_KEY')
if wandb_key:
    wandb.login(key=wandb_key)

api = wandb.Api(timeout=6000)

# Initialise empty result structure
# one row per run: experiment, model, fold, then a column per mask/metric pair
metric_rows: list = []
loss_curves_dict: dict = {}
for experiment in ['experimental_from_scratch', 'experimental_fine_tune']:
    loss_curves_dict[experiment] = {}
    for model in MODELS:
        loss_curves_dict[experiment][model] = {
            'train_loss': [],
            'val_loss': [],
            'inclusion_val_RMSE': [],
            'inclusion_val_MAE': [],
            'inclusion_val_Rel_Err': [],
            'inclusion_val_PSNR': [],
            'bg_val_RMSE': [],
            'bg_val_MAE': [],
            'bg_val_Rel_Err': [],
            'bg_val_PSNR': [],
        }


# Cache runs per project to avoid repeated API calls
project_runs: dict[str, list] = {}
# Optional: Filter by tags, state, etc. (see W&B API docs)
FILTER = {"state": "finished"}  # You can use {"state": "finished"} to get only completed runs
runs = api.runs("aisurrey_photoacoustics/MSOT_Diffusion2", filters=FILTER)



for run in runs:
    notes = run.notes
    name = run.name
    
    # if 'experimental_from_scratch' in notes or 'experimental_fine_tune' in notes:
    #     if notes.split('_')[-1] not in ['lr1em3']:
    #         print(f"Skipping run {notes} due to learning rate filter")
    #         continue
    # if notes.split('_')[-1] in ['lr1em3']:
    #     print(f"Skipping run {notes} due to learning rate filter")
    #     continue
    
    fold = int(name.split('fold')[-1])  # Assumes name format like "model_foldX"
    model = '_'.join(name.split('_')[:-1])  # Extract model name from run name
    # Assumes notes format like "experiment_model_foldX", find the string before the model name
    if f"_{model}_" in notes:
        experiment = notes.split(f"_{model}_")[0]
    else:
        raise ValueError(f"Unexpected notes format: {notes}")
    
    if model == "UNet_e2eQPAT" and run.config["no_lr_scheduler"] == True:
        print(f"Skipping run {notes} due to no_lr_scheduler==True")
        continue
    elif model == "UNet_e2eQPAT":
        print(f"detected run {notes} with no_lr_scheduler={run.config['no_lr_scheduler']}")
    
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
    print(f"Processing run: {notes}, experiment: {experiment}, model: {model}, fold: {fold}")
    metric_rows.append({
        'experiment': experiment, 'model': model, 'fold': fold,
        **{
            f'{mask}_{metric}': run.summary.get(f'{mask}_{logging_prefix}_mean_{metric}')
            for mask in mask_types for metric in METRICS
        },
    })

    # get loss curve
    if synthetic_or_experimental == 'experimental':
        for i, metric in enumerate(LOSS_CURVE_METRICS):
            loss_curves_dict_keys = list(loss_curves_dict[experiment][model].keys())
            metric_history = run.history(
                x_axis='_step', keys=[metric], stream='default', pandas=True,
            )
            loss_curves_dict[experiment][model][loss_curves_dict_keys[i]].append(metric_history[metric].tolist())

# save metrics to csv and loss curves to json
metric_columns = ['experiment', 'model', 'fold'] + [
    f'{mask}_{metric}' for mask in MASK_TYPES for metric in METRICS
]
metrics_df = pd.DataFrame(metric_rows, columns=metric_columns)
metrics_df = metrics_df.sort_values(['experiment', 'model', 'fold'])
metrics_df.to_csv('wandb_metrics.csv', index=False)
with open('wandb_loss_curves.json', 'w') as f:
    json.dump(loss_curves_dict, f, indent=4)