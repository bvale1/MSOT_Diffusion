import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"

MODELS = [
    'UNet_e2eQPAT',
    'EDM2',
    'UNet_diffusion_ablation',
]

EXPERIMENTS = [
    "experimental_from_scratch",
    "experimental_fine_tune",
]

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

MODEL_DISPLAY = {
    'UNet_e2eQPAT': 'U-Net (e2eQPAT)',
    'EDM2': 'EDM2',
    'UNet_diffusion_ablation': 'U-Net (ablation)',
}

# the original e2eQPAT reference weights are drawn as an extra green box on the
# left of the U-Net (e2eQPAT) group (they only exist for the from-scratch case)
ORIGINAL_MODEL = 'UNet_e2eQPAT_original'
ORIGINAL_EXPERIMENT = 'experimental_from_scratch'
COLOR_ORIGINAL = 'tab:green'
EXPERIMENT_COLORS = {
    'experimental_from_scratch': 'tab:blue',
    'experimental_fine_tune': 'tab:orange',
}
EXPERIMENT_DISPLAY = {
    'experimental_from_scratch': 'From scratch',
    'experimental_fine_tune': 'Fine-tuned',
}
METRIC_DISPLAY = {
    'RMSE': r'RMSE (cm$^{-1}$)',
    'MAE': r'MAE (cm$^{-1}$)',
    'Rel_Err': r'Rel. Err. (%)',
    'PSNR': 'PSNR (dB)',
}
MASK_DISPLAY = {
    'bg': 'Background',
    'inclusion': 'Inclusion',
}


def make_box_plots(
        metrics_df: pd.DataFrame,
        metric: str,
        agg: str,
        save_path: str,
    ) -> None:
    # aggregate the folds: one value per test sample per experiment/model
    value_cols = [f'{mask}_{m}' for mask in MASK_TYPES for m in METRICS]
    agg_df = metrics_df.groupby(
        ['experiment', 'model', 'sample_name']
    )[value_cols].agg(agg).reset_index()

    def get_values(model, experiment, mask):
        values = agg_df[
            (agg_df['experiment'] == experiment) & (agg_df['model'] == model)
        ][f'{mask}_{metric}'].to_numpy(dtype=float)
        return values[np.isfinite(values)]

    text_scale = 1.6
    label_fontsize = 10 * text_scale
    tick_fontsize = 8 * text_scale
    legend_fontsize = 10 * text_scale

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), layout='constrained', sharex=True)

    # one group per model; boxes within a group share a fixed pitch so they line
    # up across groups even when a group has a different number of boxes (the
    # U-Net (e2eQPAT) group additionally holds the original reference weights)
    group_spacing = 1.0
    x_positions = np.arange(len(MODELS), dtype=float) * group_spacing
    box_pitch = 0.26
    box_width = 0.22
    jitter_width = 0.06
    rng = np.random.default_rng(42)

    for row, mask in enumerate(MASK_TYPES):
        for j, model in enumerate(MODELS):
            # (values, colour) for each box in this group, ordered left to right
            group_series = []
            if model == 'UNet_e2eQPAT':
                orig_values = get_values(ORIGINAL_MODEL, ORIGINAL_EXPERIMENT, mask)
                if orig_values.size > 0:
                    group_series.append((orig_values, COLOR_ORIGINAL))
            for experiment in EXPERIMENTS:
                values = get_values(model, experiment, mask)
                if values.size > 0:
                    group_series.append((values, EXPERIMENT_COLORS[experiment]))
            if not group_series:
                continue

            k = len(group_series)
            group_offsets = (np.arange(k) - (k - 1) / 2) * box_pitch
            for (values, color), offset in zip(group_series, group_offsets):
                position = x_positions[j] + offset
                # hide the fliers, all points are scattered on top instead
                bp = ax[row].boxplot(
                    [values],
                    positions=[position],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.3},
                    whiskerprops={'color': 'black', 'linewidth': 1.0},
                    capprops={'color': 'black', 'linewidth': 1.0},
                    boxprops={'edgecolor': 'black', 'linewidth': 1.0},
                )
                for box in bp['boxes']:
                    box.set_facecolor(color)
                    box.set_alpha(0.45)
                x_jitter = position + rng.uniform(
                    -jitter_width, jitter_width, size=values.shape[0]
                )
                ax[row].scatter(
                    x_jitter,
                    values,
                    s=9,
                    color=color,
                    alpha=0.6,
                    edgecolors='none',
                    zorder=3,
                )

        ax[row].set_ylabel(
            f'{MASK_DISPLAY[mask]} {METRIC_DISPLAY.get(metric, metric)}',
            fontsize=label_fontsize,
        )
        ax[row].tick_params(axis='both', labelsize=tick_fontsize)

    ax[1].set_xlabel('Model', fontsize=label_fontsize)
    ax[1].set_xticks(x_positions)
    ax[1].set_xticklabels(
        [MODEL_DISPLAY.get(model, model) for model in MODELS], fontsize=tick_fontsize
    )

    legend_handles = [
        Patch(facecolor=COLOR_ORIGINAL, edgecolor='black', label='Original')
    ] + [
        Patch(facecolor=EXPERIMENT_COLORS[experiment], edgecolor='black',
              label=EXPERIMENT_DISPLAY[experiment])
        for experiment in EXPERIMENTS
    ]
    ax[0].legend(
        handles=legend_handles,
        ncol=len(legend_handles),
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        fontsize=legend_fontsize,
    )

    fig.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    METRIC = 'RMSE' # 'RMSE', 'MAE', 'Rel_Err' or 'PSNR'
    AGG = 'median' # aggregate the folds with 'mean' or 'median'

    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_df = pd.read_csv(os.path.join(base_dir, 'test_sample_metrics.csv'))
    save_path = os.path.join(base_dir, f'box_plot_{METRIC}_{AGG}.pdf')
    make_box_plots(metrics_df, METRIC, AGG, save_path)
    print(f'saved box plots to {save_path}')
