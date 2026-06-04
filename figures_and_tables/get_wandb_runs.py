
import os
import json
import itertools
import wandb
import numpy as np
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

METRICS = [
    'RMSE',
    'MAE',
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
sample_metrics_dict: dict = {}
for model in MODELS:
    sample_metrics_dict[model] = {}
    for metric in METRICS:
        sample_metrics_dict[model][metric] = {
            'bg': [],
            'inclusion': [],
        }

# Cache runs per project to avoid repeated API calls
project_runs: dict[str, list] = {}
# Optional: Filter by tags, state, etc. (see W&B API docs)
FILTER = {"state": "finished"}  # You can use {"state": "finished"} to get only completed runs
runs = api.runs("aisurrey_photoacoustics/MSOT_Diffusion2", filters=FILTER)

for run in runs:

    
    noise_level = run.notes
    name = run.name
    if len(name.split('_')) == 2:
        [model, input_type] = name.split('_')
    elif len(name.split('_')) == 3:
        # case model = 'Unet_e2eQPAT'
        [model1, model2, input_type] = name.split('_')
        model = f'{model1}_{model2}'
    else:
        [model1, model2, model3, input_type] = name.split('_')
        
    if model not in MODELS:
        print(f"Skipping model {model} not in MODELS list")
        continue
    if run.state != "finished":
        print(f"Skipping model {model} not finished (state={run.state})")
        continue

    artifacts = [
        a for a in run.logged_artifacts()
        if a.type == "dataset" and "test_per_sample_metrics" in a.name
    ]
    if not artifacts:
        continue
    
    # if multiple versions were logged, take latest version for this run
    artifact = max(artifacts, key=lambda a: int(a.version.lstrip("v")))

    local_dir = artifact.download(root=f"/tmp/wandb_artifacts/{run.id}")
    with open(os.path.join(local_dir, "bg.json"), "r") as f:
        bg_dict = json.load(f)
    with open(os.path.join(local_dir, "inclusion.json"), "r") as f:
        inclusion_dict = json.load(f)
    

best, worst = {}, {}
# reduces sample-level metrics to mean and percentile-based 95% CI half-width
summary_metrics_dict = deepcopy(sample_metrics_dict)
for model, metric, mask_type in itertools.product(MODELS, METRICS, MASK_TYPES):
    
            for metric, values in sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type].items():
                # values should be a list of lists [5, n_samples]
                if not values:
                    print(f"no values for {model}/{input_type}/{gt_type}/{noise_level}/{mask_type}/{metric}")
                    summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = {
                        'mean': None,
                        'ci95': None,
                    }
                    continue
                
                if metric == 'sample_names':
                    sample_names = summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric][0]
                    sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = sample_names
                    continue
                
                if len(values) != 5:
                    raise ValueError(f"Expected 5 runs for {model}/{input_type}/{gt_type}/{noise_level}/{mask_type}/{metric} but got {len(values)}")
                
                # compute mean for each sample
                metric_values = np.nanmean(np.asarray(values), axis=0)
                finite_mask = np.isfinite(metric_values)
                sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = metric_values.tolist()
                # print top and bottom three samples for this metric
                highest_idx = np.argsort(metric_values[finite_mask])[-3:]
                lowest_idx = np.argsort(metric_values[finite_mask])[:3]
                if metric in ['IoU', 'Sensitivity', 'Specificity', 'MAE', 'RMSE'] and model in ['mlp', 'segformerb5']:
                    print(f"{model}/{input_type}/{gt_type}/{noise_level}/{mask_type}/{metric}")
                    print(f"  highest: {[(sample_names[i], metric_values[i]) for i in highest_idx]}")
                    print(f"  lowest: {[(sample_names[i], metric_values[i]) for i in lowest_idx]}")
                    for i in (highest_idx if metric in ['IoU', 'Sensitivity', 'Specificity'] else lowest_idx):
                        if sample_names[i] not in best.keys():
                            best[sample_names[i]] = 1
                        else:                            
                            best[sample_names[i]] += 1
                    for i in (lowest_idx if metric in ['IoU', 'Sensitivity', 'Specificity'] else highest_idx):
                        if sample_names[i] not in worst.keys():
                            worst[sample_names[i]] = 1
                        else:                            
                            worst[sample_names[i]] += 1

                metric_values = np.nanmean(np.asarray(values), axis=1)
                valid_values = metric_values[np.isfinite(metric_values)]

                if valid_values.size == 0:
                    mean = None
                    ci95 = None
                else:
                    mean = float(np.mean(valid_values))
                    p2_5 = float(np.nanpercentile(valid_values, 2.5))
                    p97_5 = float(np.nanpercentile(valid_values, 97.5))
                    ci95 = float((p97_5 - p2_5) / 2.0)
                    
                summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = {
                    'mean': mean,
                    'ci95': ci95,
                }
                
print("Best samples (most often in top 3): ", best)
print("Worst samples (most often in bottom 3): ", worst)
# save as json
with open(os.path.join(os.path.dirname(__file__), 'per_sample_metrics.json'), 'w') as f:
    json.dump(sample_metrics_dict, f, indent=4)
with open(os.path.join(os.path.dirname(__file__), 'summary_metrics.json'), 'w') as f:
    json.dump(summary_metrics_dict, f, indent=4)