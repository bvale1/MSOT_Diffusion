import os
import numpy as np
import pandas as pd
from typing import Literal


# display names for the table
MODEL_DISPLAY = {
    'UNet_e2eQPAT_original': 'UNet (e2eQPAT, original)',
    'UNet_e2eQPAT': 'UNet (e2eQPAT)',
    'EDM2': 'EDM2',
    'UNet_diffusion_ablation': 'UNet (ablation)',
}
METRIC_DISPLAY = {
    'RMSE': r'RMSE $\downarrow$',
    'MAE': r'MAE $\downarrow$',
    'Rel_Err': r'Rel. Err. \% $\downarrow$',
    'PSNR': r'PSNR dB $\uparrow$',
}


def aggregate_folds(
    values: list,
    agg: Literal['mean_std', 'median_iqr'] = 'median_iqr',
    ) -> tuple[float, float]:
    """Aggregate a list of per-fold metric values into (centre, spread)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float('nan'), float('nan')
    if agg == 'mean_std':
        return float(np.nanmean(arr)), float(np.nanstd(arr))
    elif agg == 'median_iqr':
        centre = float(np.nanmedian(arr))
        spread = float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))
        return centre, spread
    raise ValueError(f"Unknown agg '{agg}', expected 'mean_std' or 'median_iqr'")


def build_models_dict(
    metrics_df: pd.DataFrame,
    experiment: str,
    mask: str,
    models: list,
    metrics: list,
    agg: Literal['mean_std', 'median_iqr'] = 'median_iqr',
    ) -> dict:
    """Return {model: {metric: (centre, spread)}} for one experiment/mask."""
    models_dict = {}
    for model in models:
        rows = metrics_df[
            (metrics_df['experiment'] == experiment) & (metrics_df['model'] == model)
        ]
        models_dict[model] = {}
        for metric in metrics:
            values = rows[f'{mask}_{metric}'].tolist()
            models_dict[model][metric] = aggregate_folds(values, agg)
    return models_dict


def _metric_headers(metrics: list) -> str:
    return ' & '.join(METRIC_DISPLAY.get(m, m.replace('_', ' ')) for m in metrics)


def _model_cells(models_dict: dict, model: str, metrics: list) -> str:
    cells = []
    for metric in metrics:
        centre, spread = models_dict[model][metric]
        cells.append(f"${centre:.3f} \\pm {spread:.3f}$")
    return ' & '.join(cells)


def _model_centre_cells(models_dict: dict, model: str, metrics: list) -> str:
    return ' & '.join(f"${models_dict[model][m][0]:.3f}$" for m in metrics)


def _model_spread_cells(models_dict: dict, model: str, metrics: list) -> str:
    return ' & '.join(f"$\\pm {models_dict[model][m][1]:.3f}$" for m in metrics)


def print_single_tex_reslts_table(
    models_dict: dict,
    header: str,
    metrics: list,
    caption: str,
    label: str,
    ) -> None:
    """One experiment, single ('bg') mask. Models as rows, metrics as columns."""
    n = len(metrics)
    models = list(models_dict.keys())
    col_spec = 'l|' + 'c' * n
    lines = [
        r'\begin{table}',
        r'    \centering',
        f'    \\caption{{{caption}}}',
        f'    \\label{{{label}}}',
        r'    \setlength{\tabcolsep}{4pt}',
        r'    \resizebox{\columnwidth}{!}{',
        f'    \\begin{{tabular}}{{{col_spec}}}',
        r'    \hline',
        f'    \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{{{n}}}{{c}}{{{header}}} \\\\',
        f'    \\cline{{2-{n + 1}}}',
        f'    & {_metric_headers(metrics)} \\\\',
        r'    \hline',
    ]
    for model in models:
        name = MODEL_DISPLAY.get(model, model.replace('_', ' '))
        lines.append(f'    {name} & {_model_cells(models_dict, model, metrics)} \\\\')
    lines += [
        r'    \hline',
        r'    \end{tabular}}',
        r'\end{table}',
    ]
    print('\n'.join(lines))


def print_double_tex_reslts_table(
    models_dict_left: dict, header_left: str,
    models_dict_right: dict, header_right: str,
    metrics: list,
    caption: str,
    label: str,
    ) -> None:
    """Two column-groups side by side (e.g. two experiments, both 'bg' mask).
    Models as rows, metrics repeated under each group header."""
    n = len(metrics)
    models = list(models_dict_left.keys())
    metric_headers = _metric_headers(metrics)
    col_spec = 'l|' + 'c' * n + '|' + 'c' * n
    lines = [
        r'\begin{table}[H]',
        r'    \centering',
        f'    \\caption{{{caption}}}',
        f'    \\label{{{label}}}',
        r'    \setlength{\tabcolsep}{4pt}',
        r'    \resizebox{\textwidth}{!}{',
        f'    \\begin{{tabular}}{{{col_spec}}}',
        r'    \hline',
        f'    \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{{{n}}}{{c|}}{{{header_left}}} & \\multicolumn{{{n}}}{{c}}{{{header_right}}} \\\\',
        f'    \\cline{{2-{2 * n + 1}}}',
        f'    & {metric_headers} & {metric_headers} \\\\',
        r'    \hline',
    ]
    for model in models:
        name = MODEL_DISPLAY.get(model, model.replace('_', ' '))
        row_name = '\\multirow{2}{*}{' + name + '}'
        lines.append(f'    {row_name} & {_model_centre_cells(models_dict_left, model, metrics)} & {_model_centre_cells(models_dict_right, model, metrics)} \\\\')
        lines.append(f'    & {_model_spread_cells(models_dict_left, model, metrics)} & {_model_spread_cells(models_dict_right, model, metrics)} \\\\')
        lines.append(r'    \hline')
    lines += [
        r'    \end{tabular}}',
        r'\end{table}',
    ]
    print('\n'.join(lines))


def print_blocked_tex_reslts_table(
    blocks: list,
    header_left: str,
    header_right: str,
    metrics: list,
    caption: str,
    label: str,
    ) -> None:
    """Two column-groups (left/right mask) with each experiment as a row-block.
    blocks: list of (block_name, models_dict_left, models_dict_right)."""
    n = len(metrics)
    metric_headers = _metric_headers(metrics)
    col_spec = 'l|' + 'c' * n + '|' + 'c' * n
    lines = [
        r'\begin{table}[H]',
        r'    \centering',
        f'    \\caption{{{caption}}}',
        f'    \\label{{{label}}}',
        r'    \setlength{\tabcolsep}{4pt}',
        r'    \resizebox{\textwidth}{!}{',
        f'    \\begin{{tabular}}{{{col_spec}}}',
        r'    \hline',
        f'    \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{{{n}}}{{c|}}{{{header_left}}} & \\multicolumn{{{n}}}{{c}}{{{header_right}}} \\\\',
        f'    \\cline{{2-{2 * n + 1}}}',
        f'    & {metric_headers} & {metric_headers} \\\\',
        r'    \hline',
    ]
    for block_name, models_dict_left, models_dict_right in blocks:
        lines.append(f'    \\multicolumn{{{2 * n + 1}}}{{l}}{{\\textbf{{{block_name}}}}} \\\\')
        lines.append(r'    \hline')
        for model in models_dict_left.keys():
            name = MODEL_DISPLAY.get(model, model.replace('_', ' '))
            row_name = '\\multirow{2}{*}{' + name + '}'
            lines.append(f'    {row_name} & {_model_centre_cells(models_dict_left, model, metrics)} & {_model_centre_cells(models_dict_right, model, metrics)} \\\\')
            lines.append(f'    & {_model_spread_cells(models_dict_left, model, metrics)} & {_model_spread_cells(models_dict_right, model, metrics)} \\\\')
            lines.append(r'    \hline')
    lines += [
        r'    \end{tabular}}',
        r'\end{table}',
    ]
    print('\n'.join(lines))


if __name__ == "__main__":
    MODELS = [
        'UNet_e2eQPAT',
        'EDM2',
        'UNet_diffusion_ablation',
    ]
    METRICS = [
        'RMSE',
        'MAE',
        'Rel_Err',
        'PSNR',
    ]
    # 'mean_std' for mean +/- std, 'median_iqr' for median +/- IQR
    AGG = 'median_iqr'
    #AGG = 'mean_std'

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wandb_metrics.csv')
    metrics_df = pd.read_csv(csv_path)

    # ---------------- Table 1: ImageNet_pretrain, 'bg' only ----------------
    imagenet_bg = build_models_dict(metrics_df, 'ImageNet_pretrain', 'bg', MODELS, METRICS, AGG)
    if AGG == 'mean_std':
        caption = 'Results on the synthetic ImageNet test set. Values are mean $\pm$ std of 5 folds.'
    else:
        caption = 'Results on the synthetic ImageNet test set. Values are median $\pm$ IQR of 5 folds.'
    print_single_tex_reslts_table(
        imagenet_bg,
        header='ImageNet pretrain',
        metrics=METRICS,
        caption=caption,
        label='tab:imagenet_pretrain',
    )

    # -------- Table 2: Digimouse_test | Digimouse_extrusion_test, 'bg' --------
    # digimouse_bg = build_models_dict(metrics_df, 'Digimouse_test', 'bg', MODELS, METRICS, AGG)
    # digimouse_extrusion_bg = build_models_dict(metrics_df, 'Digimouse_extrusion_test', 'bg', MODELS, METRICS, AGG)
    # if AGG == 'mean_std':
    #     caption = 'Results on the Digimouse test sets. Values are mean $\pm$ std of 5 folds.'
    # else:
    #     caption = 'Results on the Digimouse test sets. Values are median $\pm$ IQR of 5 folds.'
    # print_double_tex_reslts_table(
    #     digimouse_bg, 'Digimouse',
    #     digimouse_extrusion_bg, 'Digimouse (extrusion)',
    #     metrics=METRICS,
    #     caption=caption,
    #     label='tab:digimouse',
    # )

    # ---- Table 3: experimental_from_scratch + _fine_tune, bg | inclusion ----
    blocks = []
    for experiment, block_name, block_models in [
        ('experimental_from_scratch', 'From scratch', ['UNet_e2eQPAT_original'] + MODELS),
        ('experimental_fine_tune', 'Fine-tuned', MODELS),
    ]:
        bg = build_models_dict(metrics_df, experiment, 'bg', block_models, METRICS, AGG)
        inclusion = build_models_dict(metrics_df, experiment, 'inclusion', block_models, METRICS, AGG)
        blocks.append((block_name, bg, inclusion))
    if AGG == 'mean_std':
        caption = 'Results on the experimental phantom test set. Values are mean $\pm$ std of 5 folds.'
    else:
        caption = 'Results on the experimental phantom test set. Values are median $\pm$ IQR of 5 folds.'
    print_blocked_tex_reslts_table(
        blocks,
        header_left='Background',
        header_right='Inclusion',
        metrics=METRICS,
        caption=caption,
        label='tab:experimental',
    )
