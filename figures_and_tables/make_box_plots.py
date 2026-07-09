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
    'UNet_e2eQPAT': 'UNet (e2eQPAT)',
    'EDM2': 'EDM2',
    'UNet_diffusion_ablation': 'UNet (ablation)',
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

    text_scale = 1.6
    label_fontsize = 10 * text_scale
    tick_fontsize = 8 * text_scale
    legend_fontsize = 10 * text_scale

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), layout='constrained', sharex=True)
    colors = ['tab:blue', 'tab:orange']

    # one group per model, from-scratch and fine-tuned boxes side by side
    # within each group, with a small gap between the groups
    group_spacing = 1.0
    x_positions = np.arange(len(MODELS), dtype=float) * group_spacing
    offsets = np.linspace(-0.2, 0.2, len(EXPERIMENTS))
    box_width = 0.32
    jitter_width = 0.08
    rng = np.random.default_rng(42)

    for row, mask in enumerate(MASK_TYPES):
        for i, experiment in enumerate(EXPERIMENTS):
            data = []
            positions = []
            for j, model in enumerate(MODELS):
                values = agg_df[
                    (agg_df['experiment'] == experiment) & (agg_df['model'] == model)
                ][f'{mask}_{metric}'].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size > 0:
                    data.append(values)
                    positions.append(x_positions[j] + offsets[i])
            if not data:
                continue

            # hide the fliers, all points are scattered on top instead
            bp = ax[row].boxplot(
                data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.3},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                boxprops={'edgecolor': 'black', 'linewidth': 1.0},
            )
            for box in bp['boxes']:
                box.set_facecolor(colors[i])
                box.set_alpha(0.45)
            for x_pos, values in zip(positions, data):
                x_jitter = x_pos + rng.uniform(-jitter_width, jitter_width, size=values.shape[0])
                ax[row].scatter(
                    x_jitter,
                    values,
                    s=9,
                    color=colors[i],
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
        Patch(facecolor=colors[i], edgecolor='black', label=EXPERIMENT_DISPLAY[experiment])
        for i, experiment in enumerate(EXPERIMENTS)
    ]
    ax[0].legend(
        handles=legend_handles,
        ncol=len(EXPERIMENTS),
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
