import numpy as np
import pandas as pd
import scipy.stats as st
import wandb


def get_metrics_dict(df : pd.DataFrame, models : list, metrics_dict : dict, convert_m_to_cm : bool=True) -> dict:
    models_dict = {model : metrics_dict.copy() for model in models}
    for model in models:
        # get each fold this model was pretrained using the synthetic data
        model_df = pd.concat([
            df.loc[df['Name'] == model+'_0'],
            df.loc[df['Name'] == model+'_1'],
            df.loc[df['Name'] == model+'_2'],
            df.loc[df['Name'] == model+'_3'],
            df.loc[df['Name'] == model+'_4']
        ])
        if model_df.shape[0] != 5: # the naming convention for the models changed, so we need to check for the new naming convention
            model_df = pd.concat([
                df.loc[df['Name'] == model+'_fold0'],
                df.loc[df['Name'] == model+'_fold1'],
                df.loc[df['Name'] == model+'_fold2'],
                df.loc[df['Name'] == model+'_fold3'],
                df.loc[df['Name'] == model+'_fold4']
            ])
        if model_df.shape[0] != 5:
            model_df = df.loc[df['mean_RMSE'] != np.NaN]
        if model_df.shape[0] != 5: # there is only one fold for the digimouse datasets, although the 5 unique training folds are still there
            model_df = df.loc[df['Name'] == model+'_fold0']
        for metric in metrics_dict.keys():
            metric_values = model_df[metric].values
            #print(f'{model} {metric} {metric_values}')
            if convert_m_to_cm and metric in ['mean_RMSE', 'mean_MAE']:
                # convert from m^{-1} to cm^{-1}
                metric_values = metric_values * 1e-2
            # calculate mean and std for each metric
            models_dict[model][metric] = [np.mean(metric_values), np.std(metric_values)]
            #try:
            #    models_dict[model][metric] = [np.mean(metric_values), np.std(metric_values)]
            #except Exception as e:
            #    print(e)
            #    breakpoint()
    return models_dict

def print_single_tex_reslts_table(models_dict : dict, caption : str,
                                  label : str, metric_headers : str) -> None:
    metrics = list(list(models_dict.values())[0].keys())
    models = list(models_dict.keys())
    tex_table_string = f"""\\begin{{table*}}
    \\centering
    \\caption{caption}
    \\setlength{{\\tabcolsep}}{{3pt}}
    %\\begin{{tabular}}{{|p{{25pt}}|p{{75pt}}|p{{115pt}}|}}
    \\begin{{tabular}}{{|l|l|l|l|l|l|l|}}
    \\hline
    Model & """ + metric_headers + """ \\\\
    \\hline"""
    for model in models:
        row_name = '{' + model.replace('_',' ') + '}'
        tex_table_string += f"""\\multirow{{2}}{{*}}{row_name} & ${models_dict[model][metrics[0]][0]:.3f}$"""
        for metric in metrics[1:]:
            tex_table_string += f""" & ${models_dict[model][metric][0]:.3f}$"""
        tex_table_string += f""" \\\\ \n"""
        for metric in metrics:
            tex_table_string += f""" & $\\pm{models_dict[model][metric][1]:.3f}$"""
        tex_table_string += f""" \\\\ 
        \\hline"""
    tex_table_string += f"""\\end{{tabular}}
    \\label{label}
    \\end{{table*}}
    
    """
    
    print(tex_table_string)
    
    
def print_double_tex_reslts_table(models_dict_left : dict, header_left : str,
                                  models_dict_right : dict, header_right : str,
                                  caption : str, label : str, metric_headers : str,
                                  header_len : str='{6}') -> None:
    metrics_left = list(list(models_dict_left.values())[0].keys())
    metrics_right = list(list(models_dict_right.values())[0].keys())
    models = list(models_dict_left.keys()) # both dicts must have the same models
    tex_table_string = f"""\\begin{{table*}}
    \\centering
    \\caption{caption}
    \\label{{table}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    %\\begin{{tabular}}{{|p{{25pt}}|p{{75pt}}|p{{115pt}}|}}
    \\resizebox{{\\textwidth}}{{!}}""" + """{""" + f"""\\begin{{tabular}}""" + """{{|l|"""

    for i in range(2*int(header_len[1])):
        tex_table_string += f"""l|"""
    tex_table_string += """}}"""
    
    tex_table_string += f"""
    \\hline
    \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{header_len}{{|l|}}{header_left} & \\multicolumn{header_len}{{|l|}}{header_right} \\\\
    \\cline{{2-13}}
    & """ + metric_headers + """ & """ + metric_headers + """ \\\\
    \\hline"""
    for model in models:
        row_name = '{' + model.replace('_',' ') + '}'
        tex_table_string += f"""\\multirow{{2}}{{*}}{row_name} & ${models_dict_left[model][metrics_left[0]][0]:.3f}$"""
        for metric in metrics_left[1:]:
            tex_table_string += f""" & ${models_dict_left[model][metric][0]:.3f}$"""
        tex_table_string += f"""\n"""
        for metric in metrics_right:
            tex_table_string += f""" & ${models_dict_right[model][metric][0]:.3f}$"""
        tex_table_string += f""" \\\\ \n"""
        for metric in metrics_left:
            tex_table_string += f""" & $\\pm{models_dict_left[model][metric][1]:.3f}$"""
        tex_table_string += f"""\n"""
        for metric in metrics_right:
            tex_table_string += f""" & $\\pm{models_dict_right[model][metric][1]:.3f}$"""
        tex_table_string += f""" \\\\ 
        \\hline"""
    tex_table_string += f"""\\end{{tabular}}""" + """}""" + f"""
    \\label{label}
    \\end{{table*}}
    
    """
    
    print(tex_table_string)


api = wandb.Api()

# Optional: Filter by tags, state, etc. (see W&B API docs)
FILTER = {}  # You can use {"state": "finished"} to get only completed runs

# Project is specified by <entity/project-name>
runs = api.runs("aisurrey_photoacoustics/MSOT_Diffusion", filters=FILTER)

run_dict_list = []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    run_dict_list.append(run.summary._json_dict)
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    run_dict_list[-1].update(
        {k: v for k,v in run.config.items() if not k.startswith('_')}
    )
    # .name is the human-readable name of the run.
    run_dict_list[-1]['Name'] = run.name
    run_dict_list[-1]['Notes'] = run.notes

df = pd.DataFrame(run_dict_list)

# Preview the result
print(df.head())

columns = df.columns.values

models = [
    'UNet_e2eQPAT',
    'UNet_wl_pos_emb',
    'UNet_diffusion_ablation',
    'DDIM'
]
metrics_dict = {
    'mean_RMSE' : [None]*5, 
    'mean_MAE' : [None]*5, 
    'mean_Rel_Err' : [None]*5,
    'mean_PSNR' : [None]*5, 
    'mean_SSIM' : [None]*5
}    
metric_headers = """RMSE (cm$^{{-1}}$) & Abs. Error (cm$^{{-1}}$) & Rel. Error (\\%) & R$^{{2}}$ & PSNR & SSIM"""

print('============================== Predict Fluence ==============================')

# print TeX results table for pretraining on synthetic data
pretrain_df = df.loc[df['Notes'] == 'pretrain']
#print(f'Pretrain shape: {pretrain_df.shape}')
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, models, metrics_dict, convert_m_to_cm=True),
    '{Performance metrics for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs.}',
    '{tab:ImageNet_pretrain_metrics}',
    metric_headers
)

# print TeX results table for testing on digimouse synthetic data
digimouse_df = df.loc[df['Notes'] == 'digimouse_3d_test']
#print(f'Digimouse shape: {digimouse_df.shape}')
digimouse_extrusion_df = df.loc[df['Notes'] == 'digimouse_extrusion_test']
#print(f'Digimouse extrusion shape: {digimouse_extrusion_df.shape}')
print_double_tex_reslts_table(
    get_metrics_dict(digimouse_df, models, metrics_dict, convert_m_to_cm=True),
    '{Digimouse phantom test dataset}',
    get_metrics_dict(digimouse_extrusion_df, models, metrics_dict, convert_m_to_cm=True), 
    '{Digimouse extrusion phantom test dataset}',
    '{Performance metrics for training on the synthetic Digimouse phantom datasets. Mean and standard deviation of 5 runs.}',
    '{tab:digimouse_test_metrics}',
    metric_headers
)

# print TeX results table for testing on experimental e2eQPAT phantom data
experimental_fine_tune_df = df.loc[df['Notes'] == 'e2eQPAT_fine_tune']
#print(f'Experimental fine-tune shape: {experimental_fine_tune_df.shape}')
experimental_from_scratch_df = df.loc[df['Notes'] == 'e2eQPAT_from_scratch']
#print(f'Experimental from scratch shape: {experimental_from_scratch_df.shape}')
print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, models, metrics_dict, convert_m_to_cm=False),
    '{Fine-tune on experimental dataset}',
    get_metrics_dict(experimental_from_scratch_df, models, metrics_dict, convert_m_to_cm=False),
    '{Train from scratch on experimental dataset}',
    '{Performance metrics for training on the experimental dataset. Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}',
    metric_headers
)


print('============================== Janeks weights ==============================')
experimental_from_scratch_df = df.loc[df['Notes'] == 'e2eQPAT_Janeks_weights']
models = ['UNet_e2eQPAT']
inclusion_metrics_dict = {
    'inclusion_mean_RMSE' : [None]*5,
    'inclusion_mean_MAE' : [None]*5,
    'inclusion_mean_Rel_Err' : [None]*5,
    'inclusion_mean_PSNR' : [None]*5,
    'inclusion_mean_SSIM' : [None]*5,
}
bg_metrics_dict = {
    'mean_RMSE' : [None]*5, 
    'mean_MAE' : [None]*5, 
    'mean_Rel_Err' : [None]*5,
    'mean_PSNR' : [None]*5, 
    'mean_SSIM' : [None]*5
}
print_double_tex_reslts_table(
    get_metrics_dict(experimental_from_scratch_df, models, bg_metrics_dict, convert_m_to_cm=False),
    '{Background}',
    get_metrics_dict(experimental_from_scratch_df, models, inclusion_metrics_dict, convert_m_to_cm=False),
    '{Inclusions}',
    '{Janeks weights, I need to exceed this to publish. Performance metrics for training from scratch on the experimental dataset. Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}',
    metric_headers
)

print('============================== e2eQPAT_from_scratch_amsgrad_lrem4_eps1em8 ==============================')

experimental_fine_tune_df = df.loc[df['Notes'] == 'e2eQPAT_from_scratch_amsgrad_lrem4_eps1em8']
print(experimental_fine_tune_df.shape[0])
print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, models, bg_metrics_dict, convert_m_to_cm=False),
    '{Background}',
    get_metrics_dict(experimental_fine_tune_df, models, inclusion_metrics_dict, convert_m_to_cm=False),
    '{Inclusions}',
    '{amsgrad, lr 1e-4, eps 1e-8. Performance metrics for training from scratch on the experimental dataset. Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}',
    metric_headers
)

print('============================== e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8 ==============================')

models = ['UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
experimental_fine_tune_df = df.loc[df['Notes'] == 'e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8']
print(experimental_fine_tune_df.shape[0])
print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, models, bg_metrics_dict, convert_m_to_cm=False),
    '{Background}',
    get_metrics_dict(experimental_fine_tune_df, models, inclusion_metrics_dict, convert_m_to_cm=False),
    '{Inclusions}',
    '{amsgrad, lr 1e-4, eps 1e-8. Performance metrics for fine-tuning on the experimental dataset. Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}',
    metric_headers
)

print('============================== pretrain_lr1em3_eps1em8 ==============================')

models = ['UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
bg_metrics_dict = {
    'bg_synthetic_test_mean_RMSE' : [None]*5,
    'bg_synthetic_test_mean_MAE' : [None]*5,
    'bg_synthetic_test_mean_Rel_Err' : [None]*5,
    'bg_synthetic_test_mean_PSNR' : [None]*5,
    'bg_synthetic_test_mean_SSIM' : [None]*5
}
pretrain_df = df.loc[df['Notes'] == 'pretrain_lr1em3_eps1em8']
print(pretrain_df.shape[0])
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, models, bg_metrics_dict, convert_m_to_cm=True),
    '{Pretrain, lr 1e-3, eps 1e-8. Performance metrics for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs.}',
    '{tab:ImageNet_pretrain_metrics}',
    metric_headers
)

print('============================== e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8 ==============================')

models = ['UNet_e2eQPAT']
metric_headers = """RMSE (cm$^{{-1}}$) & Rel. Error (\\%) & PSNR & val loss/train loss"""
bg_metrics_dict = {
    'bg_experimental_test_mean_RMSE' : [None]*5,
    'bg_experimental_test_mean_Rel_Err' : [None]*5,
    'bg_experimental_test_mean_PSNR' : [None]*5
}
inclusion_metrics_dict = {
    'inclusion_experimental_test_mean_RMSE' : [None]*5,
    'inclusion_experimental_test_mean_Rel_Err' : [None]*5,
    'inclusion_experimental_test_mean_PSNR' : [None]*5
}
breakpoint()
bg_model_dict = get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_Janeks_weights'], models, bg_metrics_dict, convert_m_to_cm=False
)
bg_model_dict['UNet_e2eQPAT_not_pretrained'] = get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_from_scratch_amsgrad_lrem4_eps1em8'], models, bg_metrics_dict, convert_m_to_cm=False
)['UNet_e2eQPAT']
bg_model_dict['UNet_e2eQPAT_fine_tune'] = get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8'], models, bg_metrics_dict, convert_m_to_cm=False
)['UNet_e2eQPAT']

inclusion_model_dict = get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_Janeks_weights'], models, inclusion_metrics_dict, convert_m_to_cm=False
)
inclusion_model_dict['UNet_e2eQPAT_not_pretrained'] = get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_from_scratch_amsgrad_lrem4_eps1em8'], models, inclusion_metrics_dict, convert_m_to_cm=False
)['UNet_e2eQPAT']
inclusion_model_dict['UNet_e2eQPAT_fine_tune'] = get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8'], models, inclusion_metrics_dict, convert_m_to_cm=False
)['UNet_e2eQPAT']

models = ['UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
bg_model_dict.update(get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8'], models, bg_metrics_dict, convert_m_to_cm=False
))
inclusion_model_dict.update(get_metrics_dict(
    df.loc[df['Notes'] == 'e2eQPAT_fine_tune_amsgrad_lr1em4_eps1em8'], models, inclusion_metrics_dict, convert_m_to_cm=False
))

print_double_tex_reslts_table(
    bg_model_dict,
    '{Background}',
    inclusion_model_dict,
    '{Inclusions}',
    '{amsgrad, lr 1e-4, eps 1e-8. Performance metrics for fine-tuning on experimental dataset. Checkpoints trained with eps 1e-8, Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}',
    metric_headers,
    header_len='{4}'
)

print('============================== unet_wl_pos_emb_synthetic_noAttn ==============================')
metric_headers = """RMSE (cm$^{{-1}}$) & Abs. Error (cm$^{{-1}}$) & Rel. Error (\\%) & PSNR & SSIM & val loss/train loss"""
#models = ['UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
models = ['UNet_wl_pos_emb']
bg_metrics_dict = {
    'bg_synthetic_test_mean_RMSE' : [None]*5,
    'bg_synthetic_test_mean_MAE' : [None]*5,
    'bg_synthetic_test_mean_Rel_Err' : [None]*5,
    'bg_synthetic_test_mean_PSNR' : [None]*5,
    'bg_synthetic_test_mean_SSIM' : [None]*5
}

pretrain_df = df.loc[df['Notes'] == 'pretrain_lr1em3_eps1em8_noAttn']
print(pretrain_df.shape[0])
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, models, bg_metrics_dict, convert_m_to_cm=True),
    '{Pretrain, lr 1e-3, eps 1e-8, noAttn. Performance metrics for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs.}',
    '{tab:ImageNet_pretrain_metrics}',
    metric_headers
)

print('============================== e2eQPAT_fine_tune_lr1em4_eps1em8_noAttn ==============================')

metric_headers = """RMSE (cm$^{{-1}}$) & Rel. Error (\\%) & PSNR & val loss/train loss"""
#models = ['UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
models = ['UNet_wl_pos_emb']
bg_metrics_dict = {
    'bg_experimental_test_mean_RMSE' : [None]*5,
    'bg_experimental_test_mean_Rel_Err' : [None]*5,
    'bg_experimental_test_mean_PSNR' : [None]*5
}
inclusion_metrics_dict = {
    'inclusion_experimental_test_mean_RMSE' : [None]*5,
    'inclusion_experimental_test_mean_Rel_Err' : [None]*5,
    'inclusion_experimental_test_mean_PSNR' : [None]*5
}
pretrain_df = df.loc[df['Notes'] == 'e2eQPAT_fine_tune_noAttn_lr1em4_eps1em8']

print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, models, bg_metrics_dict, convert_m_to_cm=False),
    '{Background}',
    get_metrics_dict(experimental_fine_tune_df, models, inclusion_metrics_dict, convert_m_to_cm=False),
    '{Inclusions}',
    '{fine tune, noAttn, lr1e-4.}',
    '{tab:experimental_test_metrics}',
    metric_headers, header_len='{4}'
)

print('============================== pretrain_timeCondUnet_lr1e-3 ==============================')
metric_headers = """RMSE (cm$^{{-1}}$) & Rel. Error (\\%) & PSNR & val loss/train loss"""
#models = ['UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
models = ['UNet_wl_pos_emb', 'DDIM']
bg_metrics_dict = {
    'bg_synthetic_test_mean_RMSE' : [None]*5,
    'bg_synthetic_test_mean_Rel_Err' : [None]*5,
    'bg_synthetic_test_mean_PSNR' : [None]*5
}

pretrain_df = df.loc[df['Notes'] == 'pretrain_tCondUnet']
print(pretrain_df.shape[0])
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, models, bg_metrics_dict, convert_m_to_cm=True),
    '{Pretrain, lr 1e-3, timeCondUnet.}',
    '{tab:ImageNet_pretrain_metrics}',
    metric_headers
)

print('============================== fine_timeCondUnet_lr1e-4 ==============================')

metric_headers = """RMSE (cm$^{{-1}}$) & Rel. Error (\\%) & PSNR & val loss/train loss"""
#models = ['UNet_e2eQPAT', 'UNet_wl_pos_emb', 'UNet_diffusion_ablation', 'DDIM']
models = ['UNet_wl_pos_emb', 'DDIM']
bg_metrics_dict = {
    'bg_experimental_test_mean_RMSE' : [None]*5,
    'bg_experimental_test_mean_Rel_Err' : [None]*5,
    'bg_experimental_test_mean_PSNR' : [None]*5
}
inclusion_metrics_dict = {
    'inclusion_experimental_test_mean_RMSE' : [None]*5,
    'inclusion_experimental_test_mean_Rel_Err' : [None]*5,
    'inclusion_experimental_test_mean_PSNR' : [None]*5
}
pretrain_df = df.loc[df['Notes'] == 'e2eQPAT_fine_tune_tCondUnet']

print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, models, bg_metrics_dict, convert_m_to_cm=False),
    '{Background}',
    get_metrics_dict(experimental_fine_tune_df, models, inclusion_metrics_dict, convert_m_to_cm=False),
    '{Inclusions}',
    '{fine tune timeCondUnet, lr1e-4.}',
    '{tab:experimental_test_metrics}',
    metric_headers, header_len='{4}'
)