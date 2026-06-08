import os
import json
import numpy as np
from typing import Literal


# display names for the table
MODEL_DISPLAY = {
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
    metrics_json: dict,
    experiment: str,
    mask: str,
    models: list,
    metrics: list,
    agg: Literal['mean_std', 'median_iqr'] = 'median_iqr',
    ) -> dict:
    """Return {model: {metric: (centre, spread)}} for one experiment/mask."""
    models_dict = {}
    for model in models:
        models_dict[model] = {}
        for metric in metrics:
            values = metrics_json[experiment][model][metric][mask]
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
        r'\begin{table*}',
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
        left = _model_cells(models_dict_left, model, metrics)
        right = _model_cells(models_dict_right, model, metrics)
        lines.append(f'    {name} & {left} & {right} \\\\')
    lines += [
        r'    \hline',
        r'    \end{tabular}}',
        r'\end{table*}',
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
        r'\begin{table*}',
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
            left = _model_cells(models_dict_left, model, metrics)
            right = _model_cells(models_dict_right, model, metrics)
            lines.append(f'    {name} & {left} & {right} \\\\')
        lines.append(r'    \hline')
    lines += [
        r'    \end{tabular}}',
        r'\end{table*}',
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

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wandb_metrics.json')
    with open(json_path, 'r') as f:
        metrics_json = json.load(f)

    # ---------------- Table 1: ImageNet_pretrain, 'bg' only ----------------
    imagenet_bg = build_models_dict(metrics_json, 'ImageNet_pretrain', 'bg', MODELS, METRICS, AGG)
    print_single_tex_reslts_table(
        imagenet_bg,
        header='ImageNet pretrain (background)',
        metrics=METRICS,
        caption='Results on the synthetic ImageNet test set.',
        label='tab:imagenet_pretrain',
    )

    # -------- Table 2: Digimouse_test | Digimouse_extrusion_test, 'bg' --------
    digimouse_bg = build_models_dict(metrics_json, 'Digimouse_test', 'bg', MODELS, METRICS, AGG)
    digimouse_extrusion_bg = build_models_dict(metrics_json, 'Digimouse_extrusion_test', 'bg', MODELS, METRICS, AGG)
    print_double_tex_reslts_table(
        digimouse_bg, 'Digimouse',
        digimouse_extrusion_bg, 'Digimouse (extrusion)',
        metrics=METRICS,
        caption='Out-of-distribution results on the Digimouse test sets (background).',
        label='tab:digimouse',
    )

    # ---- Table 3: experimental_from_scratch + _fine_tune, bg | inclusion ----
    blocks = []
    for experiment, block_name in [
        ('experimental_from_scratch', 'From scratch'),
        ('experimental_fine_tune', 'Fine-tuned'),
    ]:
        bg = build_models_dict(metrics_json, experiment, 'bg', MODELS, METRICS, AGG)
        inclusion = build_models_dict(metrics_json, experiment, 'inclusion', MODELS, METRICS, AGG)
        blocks.append((block_name, bg, inclusion))
    print_blocked_tex_reslts_table(
        blocks,
        header_left='Background',
        header_right='Inclusion',
        metrics=METRICS,
        caption='Results on the experimental phantom test set.',
        label='tab:experimental',
    )
