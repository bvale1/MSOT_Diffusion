import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
from itertools import product

# number of training epochs per model, used to scale the x-axis into epochs
EPOCHS = {
    'UNet_e2eQPAT': 200,
    'EDM2': 1000,
    'UNet_diffusion_ablation': 1000,
}


def _stack_folds(fold_curves: list) -> np.ndarray:
    """Stack a list of variable-length per-fold curves into a (n_folds, max_len)
    array, padding shorter folds with NaN."""
    if len(fold_curves) < 5:
        print(f"Warning: only {len(fold_curves)} folds found, expected 5")
    fold_curves = [np.asarray(f, dtype=np.float32) for f in fold_curves if len(f) > 0]
    max_len = max(len(f) for f in fold_curves)
    stacked = np.full((len(fold_curves), max_len), np.nan, dtype=np.float32)
    for i, f in enumerate(fold_curves):
        stacked[i, :len(f)] = f
    return stacked


def plot_loss_curves(
    loss_curves: dict,
    model: str,
    experiments: list,
    labels: list,
    savename: str,
    metric1_key: list[str] = ['train_loss'],
    metric2_key: list[str] = ['val_loss'],
    std_or_iqr: Literal['std', 'iqr'] = 'std',
    linthresh: None | float = None,
    ) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if linthresh is not None:
        ax.set_yscale('symlog', linthresh=linthresh)
    else:
        ax.set_yscale('log')
    ylabel = r'Mean squared error (a.u.)'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epochs')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    epochs = EPOCHS[model]

    for i, experiment in enumerate(experiments):
        model_curves = loss_curves[experiment][model]
        if len(model_curves['train_loss']) == 0:
            print(f"No loss curves for {experiment}/{model}, skipping")
            continue

        metric1 = _stack_folds(model_curves[metric1_key])
        if metric2_key is not None:
            metric2 = _stack_folds(model_curves[metric2_key])
        else:
            metric2 = np.full((metric1.shape[0], 1), np.nan, dtype=np.float32)

        # metrics are logged at uniform intervals, so map each onto [0, epochs]
        metric1_x = np.linspace(0, epochs, metric1.shape[1])
        metric2_x = np.linspace(0, epochs, metric2.shape[1])

        if std_or_iqr == 'std':
            metric1_centre = np.nanmean(metric1, axis=0)
            metric2_centre = np.nanmean(metric2, axis=0) if metric2_key is not None else None
            metric1_lo = metric1_centre - np.nanstd(metric1, axis=0)
            metric1_hi = metric1_centre + np.nanstd(metric1, axis=0)
            metric2_lo = metric2_centre - np.nanstd(metric2, axis=0) if metric2_key is not None else None
            metric2_hi = metric2_centre + np.nanstd(metric2, axis=0) if metric2_key is not None else None
        elif std_or_iqr == 'iqr':
            metric1_centre = np.nanmedian(metric1, axis=0)
            metric2_centre = np.nanmedian(metric2, axis=0) if metric2_key is not None else None
            metric1_lo = np.nanpercentile(metric1, 25, axis=0)
            metric1_hi = np.nanpercentile(metric1, 75, axis=0)
            metric2_lo = np.nanpercentile(metric2, 25, axis=0) if metric2_key is not None else None
            metric2_hi = np.nanpercentile(metric2, 75, axis=0) if metric2_key is not None else None

        ax.plot(metric1_x, metric1_centre, color=colors[i], linestyle='-', label=f'{labels[i]} {metric1_key}')
        ax.fill_between(metric1_x, metric1_lo, metric1_hi, color=colors[i], alpha=0.2)
        if metric2_key is not None:
            ax.plot(metric2_x, metric2_centre, color=colors[i], linestyle='--', label=f'{labels[i]} {metric2_key}')
            ax.fill_between(metric2_x, metric2_lo, metric2_hi, color=colors[i], alpha=0.2)

    ax.legend(loc='upper right')
    ax.set_xlim(0, epochs)
    #ax.grid(True)
    #ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(savename, dpi=300, bbox_inches='tight', format='pdf')


if __name__ == "__main__":
    # choose the model and compare it across the two experiments
    MODELS = ['UNet_e2eQPAT', 'EDM2']
    LINTHRESH = {'UNet_e2eQPAT': None, 'EDM2': None}
    EXPERIMENTS = ['experimental_from_scratch', 'experimental_fine_tune']
    LABELS = ['not pretrained', 'pretrained']
    # primary loss curve figure per model. EDM2 is trained with the EDM2 denoising
    # objective, whose loss can be negative and is not the same quantity as the MSE
    # validation loss, so only its validation loss is plotted (no train_loss).
    MAIN_METRIC_KEYS = {
        'UNet_e2eQPAT': ['train_loss', 'val_loss'],
        'EDM2': ['val_loss', None],
    }
    METIC_KEYS = [
        ['inclusion_val_RMSE', None],
        ['inclusion_val_MAE', None],
        ['inclusion_val_Rel_Err', None],
        ['bg_val_RMSE', None],
        ['bg_val_MAE', None],
        ['bg_val_Rel_Err', None],
    ]

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wandb_loss_curves.json')
    with open(json_path, 'r') as f:
        loss_curves = json.load(f)

    for model in MODELS:
        metric_keys = MAIN_METRIC_KEYS[model]
        savename = f'loss_curves_{model}_{metric_keys[0]}_{EXPERIMENTS[0]}_vs_{EXPERIMENTS[1]}.pdf'
        plot_loss_curves(
            loss_curves=loss_curves,
            model=model,
            experiments=EXPERIMENTS,
            labels=LABELS,
            savename=savename,
            metric1_key=metric_keys[0],
            metric2_key=metric_keys[1],
            std_or_iqr='iqr',
            linthresh=LINTHRESH[model],
        )

    for model, metric_keys in product(MODELS, METIC_KEYS):
        savename = f'loss_curves_{model}_{metric_keys[0]}_{EXPERIMENTS[0]}_vs_{EXPERIMENTS[1]}.pdf'
        plot_loss_curves(
            loss_curves=loss_curves,
            model=model,
            experiments=EXPERIMENTS,
            labels=LABELS,
            savename=savename,
            metric1_key=metric_keys[0],
            metric2_key=metric_keys[1],
            std_or_iqr='iqr',
            linthresh=LINTHRESH[model],
        )
