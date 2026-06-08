import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal


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
    std_or_iqr: Literal['std', 'iqr'] = 'std',
    ) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_yscale('log')
    ylabel = r'Denoising loss (a.u.)' if model == 'EDM2' else r'Mean squared error (a.u.)'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epochs')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    epochs = EPOCHS[model]

    for i, experiment in enumerate(experiments):
        model_curves = loss_curves[experiment][model]
        if len(model_curves['train_loss']) == 0:
            print(f"No loss curves for {experiment}/{model}, skipping")
            continue

        train_loss = _stack_folds(model_curves['train_loss'])
        val_loss = _stack_folds(model_curves['val_loss'])

        # train/val are logged at uniform intervals, so map each onto [0, epochs]
        train_x = np.linspace(0, epochs, train_loss.shape[1])
        val_x = np.linspace(0, epochs, val_loss.shape[1])

        if std_or_iqr == 'std':
            train_centre = np.nanmean(train_loss, axis=0)
            val_centre = np.nanmean(val_loss, axis=0)
            train_lo = train_centre - np.nanstd(train_loss, axis=0)
            train_hi = train_centre + np.nanstd(train_loss, axis=0)
            val_lo = val_centre - np.nanstd(val_loss, axis=0)
            val_hi = val_centre + np.nanstd(val_loss, axis=0)
        elif std_or_iqr == 'iqr':
            train_centre = np.nanmedian(train_loss, axis=0)
            val_centre = np.nanmedian(val_loss, axis=0)
            train_lo = np.nanpercentile(train_loss, 25, axis=0)
            train_hi = np.nanpercentile(train_loss, 75, axis=0)
            val_lo = np.nanpercentile(val_loss, 25, axis=0)
            val_hi = np.nanpercentile(val_loss, 75, axis=0)

        ax.plot(train_x, train_centre, color=colors[i], linestyle='-', label=f'{labels[i]} train loss')
        ax.plot(val_x, val_centre, color=colors[i], linestyle='--', label=f'{labels[i]} validation loss')
        ax.fill_between(train_x, train_lo, train_hi, color=colors[i], alpha=0.2)
        ax.fill_between(val_x, val_lo, val_hi, color=colors[i], alpha=0.2)

    ax.legend(loc='upper right')
    ax.set_xlim(0, epochs)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(savename, dpi=300, bbox_inches='tight', format='pdf')


if __name__ == "__main__":
    # choose the model and compare it across the two experiments
    MODEL = 'UNet_e2eQPAT'
    EXPERIMENTS = ['experimental_from_scratch', 'experimental_fine_tune']
    LABELS = ['not pretrained', 'pretrained']

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wandb_loss_curves.json')
    with open(json_path, 'r') as f:
        loss_curves = json.load(f)

    savename = f'loss_curves_{MODEL}_{EXPERIMENTS[0]}_vs_{EXPERIMENTS[1]}.pdf'
    plot_loss_curves(
        loss_curves, MODEL, EXPERIMENTS, LABELS, savename, std_or_iqr='std'
    )
