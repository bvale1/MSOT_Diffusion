import numpy as np
import pandas as pd
import scipy.stats as st

def get_metrics_dict(df : pd.DataFrame, convert_m_to_cm : bool=True) -> dict:
    models = [
        'UNet_e2eQPAT',
        'UNet_wl_pos_emb',
        'UNet_diffusion_ablation',
        'DDIM'
    ]
    metrics_dict = {
        'mean_RMSE (Min)' : [None]*5, 
        'mean_MAE' : [None]*5, 
        'mean_Rel_Err (Min)' : [None]*5,
        'mean_R2' : [None]*5, 
        'mean_PSNR (Min)' : [None]*5, 
        'mean_SSIM (Min)' : [None]*5
    }    
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
        if model_df.shape[0] != 5: # there is only one fold for the digimouse datasets, although the 5 unique training folds are still there
            model_df = df.loc[df['Name'] == model+'_fold0']
        for metric in metrics_dict.keys():
            metric_values = model_df[metric].values
            #print(f'{model} {metric} {metric_values}')
            if convert_m_to_cm and metric in ['mean_RMSE (Min)', 'mean_MAE']:
                # convert from m^{-1} to cm^{-1}
                metric_values = metric_values * 1e-2
            # calculate mean and std for each metric
            models_dict[model][metric] = [np.mean(metric_values), np.std(metric_values)]
        
    return models_dict

def print_single_tex_reslts_table(models_dict : dict, caption : str, label : str) -> None:
    print(f"""\\begin{{table*}}
    \\centering
    \\caption{caption}
    \\label{{table}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    %\\begin{{tabular}}{{|p{{25pt}}|p{{75pt}}|p{{115pt}}|}}
    \\begin{{tabular}}{{|l|l|l|l|l|l|l|}}
    \\hline
    Model & RMSE (cm$^{{-1}}$) & Abs. Error (cm$^{{-1}}$) & Rel. Error (\\%) & R$^{{2}}$ & PSNR & SSIM \\\\
    \\hline
    \\multirow{{2}}{{*}}{{U-Net}} & ${models_dict['UNet_e2eQPAT']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict['UNet_e2eQPAT']['mean_MAE'][0]:.3f}$
        & ${models_dict['UNet_e2eQPAT']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict['UNet_e2eQPAT']['mean_R2'][0]:.3f}$
        & ${models_dict['UNet_e2eQPAT']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict['UNet_e2eQPAT']['mean_SSIM (Min)'][0]:.3f}$ \\\\
    
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\multirow{{2}}{{*}}{{WL Pos Emb}} & ${models_dict['UNet_wl_pos_emb']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict['UNet_wl_pos_emb']['mean_MAE'][0]:.3f}$
        & ${models_dict['UNet_wl_pos_emb']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict['UNet_wl_pos_emb']['mean_R2'][0]:.3f}$
        & ${models_dict['UNet_wl_pos_emb']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict['UNet_wl_pos_emb']['mean_SSIM (Min)'][0]:.3f}$ \\\\
    
        & $\\pm{models_dict['UNet_wl_pos_emb']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_wl_pos_emb']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict['UNet_wl_pos_emb']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_wl_pos_emb']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict['UNet_wl_pos_emb']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_wl_pos_emb']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\multirow{{2}}{{*}}{{Diffusion Ablation}} & ${models_dict['UNet_diffusion_ablation']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict['UNet_diffusion_ablation']['mean_MAE'][0]:.3f}$
        & ${models_dict['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict['UNet_diffusion_ablation']['mean_R2'][0]:.3f}$
        & ${models_dict['UNet_diffusion_ablation']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict['UNet_diffusion_ablation']['mean_SSIM (Min)'][0]:.3f}$ \\\\
        
        & $\\pm{models_dict['UNet_diffusion_ablation']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_diffusion_ablation']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_diffusion_ablation']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict['UNet_diffusion_ablation']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_diffusion_ablation']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\multirow{{2}}{{*}}{{Conditional Diffusion}} & ${models_dict['DDIM']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict['DDIM']['mean_MAE'][0]:.3f}$
        & ${models_dict['DDIM']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict['DDIM']['mean_R2'][0]:.3f}$
        & ${models_dict['DDIM']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict['DDIM']['mean_SSIM (Min)'][0]:.3f}$ \\\\
    
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict['UNet_e2eQPAT']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\end{{tabular}}
    \\label{label}
    \\end{{table*}}
    """)
    
def print_double_tex_reslts_table(models_dict_left : dict, header_left : str,
                                  models_dict_right : dict, header_right : str,
                                  caption : str, label : str) -> None:
    print(f"""\\begin{{table*}}
    \\centering
    \\caption{caption}
    \\label{{table}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    %\\begin{{tabular}}{{|p{{25pt}}|p{{75pt}}|p{{115pt}}|}}
    \\begin{{tabular}}{{|l|l|l|l|l|l|l|l|l|l|l|l|l|}}
    \\hline
    \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{{6}}{{|l|}}{header_left} & \\multicolumn{{6}}{{|l|}}{header_right} \\\\
    \\cline{{2-13}}
    & RMSE (cm$^{{-1}}$) & Abs. Error (cm$^{{-1}}$) & Rel. Error (\\%) & R$^{{2}}$ & PSNR & SSIM & RMSE (cm$^{{-1}}$) & Abs. Error (cm$^{{-1}}$) & Rel. Error (\\%) & R$^{{2}}$ & PSNR & SSIM \\\\
    \\hline
    \\multirow{{2}}{{*}}{{U-Net}} & ${models_dict_left['UNet_e2eQPAT']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_e2eQPAT']['mean_MAE'][0]:.3f}$
        & ${models_dict_left['UNet_e2eQPAT']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_e2eQPAT']['mean_R2'][0]:.3f}$
        & ${models_dict_left['UNet_e2eQPAT']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_e2eQPAT']['mean_SSIM (Min)'][0]:.3f}$
        
        & ${models_dict_right['UNet_e2eQPAT']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_e2eQPAT']['mean_MAE'][0]:.3f}$
        & ${models_dict_right['UNet_e2eQPAT']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_e2eQPAT']['mean_R2'][0]:.3f}$
        & ${models_dict_right['UNet_e2eQPAT']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_e2eQPAT']['mean_SSIM (Min)'][0]:.3f}$ \\\\
        
        & $\\pm{models_dict_left['UNet_e2eQPAT']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_e2eQPAT']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_e2eQPAT']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_e2eQPAT']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_e2eQPAT']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_e2eQPAT']['mean_SSIM (Min)'][1]:.3f}$
        
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\multirow{{2}}{{*}}{{WL Pos Emb}} & ${models_dict_left['UNet_wl_pos_emb']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_wl_pos_emb']['mean_MAE'][0]:.3f}$
        & ${models_dict_left['UNet_wl_pos_emb']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_wl_pos_emb']['mean_R2'][0]:.3f}$
        & ${models_dict_left['UNet_wl_pos_emb']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_wl_pos_emb']['mean_SSIM (Min)'][0]:.3f}$
        
        & ${models_dict_right['UNet_wl_pos_emb']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_wl_pos_emb']['mean_MAE'][0]:.3f}$
        & ${models_dict_right['UNet_wl_pos_emb']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_wl_pos_emb']['mean_R2'][0]:.3f}$
        & ${models_dict_right['UNet_wl_pos_emb']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_wl_pos_emb']['mean_SSIM (Min)'][0]:.3f}$ \\\\
        
        & $\\pm{models_dict_left['UNet_wl_pos_emb']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_wl_pos_emb']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_wl_pos_emb']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_wl_pos_emb']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_wl_pos_emb']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_wl_pos_emb']['mean_SSIM (Min)'][1]:.3f}$
        
        & $\\pm{models_dict_right['UNet_wl_pos_emb']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_wl_pos_emb']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_wl_pos_emb']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_wl_pos_emb']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_wl_pos_emb']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_wl_pos_emb']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\multirow{{2}}{{*}}{{Diffusion Ablation}} & ${models_dict_left['UNet_diffusion_ablation']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_diffusion_ablation']['mean_MAE'][0]:.3f}$
        & ${models_dict_left['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_diffusion_ablation']['mean_R2'][0]:.3f}$
        & ${models_dict_left['UNet_diffusion_ablation']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_left['UNet_diffusion_ablation']['mean_SSIM (Min)'][0]:.3f}$
        
        & ${models_dict_right['UNet_diffusion_ablation']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_diffusion_ablation']['mean_MAE'][0]:.3f}$
        & ${models_dict_right['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_diffusion_ablation']['mean_R2'][0]:.3f}$
        & ${models_dict_right['UNet_diffusion_ablation']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_right['UNet_diffusion_ablation']['mean_SSIM (Min)'][0]:.3f}$ \\\\
        
        & $\\pm{models_dict_left['UNet_diffusion_ablation']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_diffusion_ablation']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_diffusion_ablation']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_diffusion_ablation']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['UNet_diffusion_ablation']['mean_SSIM (Min)'][1]:.3f}$
        
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['UNet_diffusion_ablation']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\multirow{{2}}{{*}}{{Conditional Diffusion}} & ${models_dict_left['DDIM']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_left['DDIM']['mean_MAE'][0]:.3f}$
        & ${models_dict_left['DDIM']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_left['DDIM']['mean_R2'][0]:.3f}$
        & ${models_dict_left['DDIM']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_left['DDIM']['mean_SSIM (Min)'][0]:.3f}$
        
        & ${models_dict_right['DDIM']['mean_RMSE (Min)'][0]:.3f}$
        & ${models_dict_right['DDIM']['mean_MAE'][0]:.3f}$
        & ${models_dict_right['DDIM']['mean_Rel_Err (Min)'][0]:.3f}$
        & ${models_dict_right['DDIM']['mean_R2'][0]:.3f}$
        & ${models_dict_right['DDIM']['mean_PSNR (Min)'][0]:.3f}$
        & ${models_dict_right['DDIM']['mean_SSIM (Min)'][0]:.3f}$ \\\\
        
        & $\\pm{models_dict_left['DDIM']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['DDIM']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_left['DDIM']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['DDIM']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_left['DDIM']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_left['DDIM']['mean_SSIM (Min)'][1]:.3f}$
        
        & $\\pm{models_dict_right['DDIM']['mean_RMSE (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['DDIM']['mean_MAE'][1]:.3f}$
        & $\\pm{models_dict_right['DDIM']['mean_Rel_Err (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['DDIM']['mean_R2'][1]:.3f}$
        & $\\pm{models_dict_right['DDIM']['mean_PSNR (Min)'][1]:.3f}$
        & $\\pm{models_dict_right['DDIM']['mean_SSIM (Min)'][1]:.3f}$ \\\\
    \\hline
    \\end{{tabular}}
    \\label{label}
    \\end{{table*}}
    """)
    
# load dataframe and convert to numpy array
file = 'wandb_export_2025-04-28T12_31_21.033+01_00.csv'
df = pd.read_csv(file)
columns = df.columns.values

print('============================== Predict Fluence ==============================')
'''
# print TeX results table for pretraining on synthetic data
pretrain_df = df.loc[df['Notes'] == 'pretrain']
#print(f'Pretrain shape: {pretrain_df.shape}')
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, convert_m_to_cm=True),
    '{Performance metrics for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs.}',
    '{tab:ImageNet_pretrain_metrics}'
)

# print TeX results table for testing on digimouse synthetic data
digimouse_df = df.loc[df['Notes'] == 'digimouse_3d_test']
#print(f'Digimouse shape: {digimouse_df.shape}')
digimouse_extrusion_df = df.loc[df['Notes'] == 'digimouse_extrusion_test']
#print(f'Digimouse extrusion shape: {digimouse_extrusion_df.shape}')
print_double_tex_reslts_table(
    get_metrics_dict(digimouse_df, convert_m_to_cm=True),
    '{Digimouse phantom test dataset}',
    get_metrics_dict(digimouse_extrusion_df, convert_m_to_cm=True), 
    '{Digimouse extrusion phantom test dataset}',
    '{Performance metrics for training on the synthetic Digimouse phantom datasets. Mean and standard deviation of 5 runs.}',
    '{tab:digimouse_test_metrics}'
)

# print TeX results table for testing on experimental e2eQPAT phantom data
experimental_fine_tune_df = df.loc[df['Notes'] == 'e2eQPAT_fine_tune']
#print(f'Experimental fine-tune shape: {experimental_fine_tune_df.shape}')
experimental_from_scratch_df = df.loc[df['Notes'] == 'e2eQPAT_from_scratch']
#print(f'Experimental from scratch shape: {experimental_from_scratch_df.shape}')
print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, convert_m_to_cm=False),
    '{Fine-tune on experimental dataset}',
    get_metrics_dict(experimental_from_scratch_df, convert_m_to_cm=False),
    '{Train from scratch on experimental dataset}',
    '{Performance metrics for training on the experimental dataset. Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}'
)
'''
print('============================== No Fluence ==============================')

# print TeX results table for pretraining on synthetic data
pretrain_df = df.loc[df['Notes'] == 'pretrain_no_fluence']
#print(f'Pretrain shape: {pretrain_df.shape}')
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, convert_m_to_cm=True),
    '{Performance metrics for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs.}',
    '{tab:ImageNet_pretrain_metrics}'
)

# print TeX results table for testing on digimouse synthetic data
digimouse_df = df.loc[df['Notes'] == 'digimouse_3d_test']
#print(f'Digimouse shape: {digimouse_df.shape}')
digimouse_extrusion_df = df.loc[df['Notes'] == 'digimouse_extrusion_test']
#print(f'Digimouse extrusion shape: {digimouse_extrusion_df.shape}')
print_double_tex_reslts_table(
    get_metrics_dict(digimouse_df, convert_m_to_cm=True),
    '{Digimouse phantom test dataset}',
    get_metrics_dict(digimouse_extrusion_df, convert_m_to_cm=True), 
    '{Digimouse extrusion phantom test dataset}',
    '{Performance metrics for training on the synthetic Digimouse phantom datasets. Mean and standard deviation of 5 runs.}',
    '{tab:digimouse_test_metrics}'
)

# print TeX results table for testing on experimental e2eQPAT phantom data
experimental_fine_tune_df = df.loc[df['Notes'] == 'e2eQPAT_fine_tune']
#print(f'Experimental fine-tune shape: {experimental_fine_tune_df.shape}')
experimental_from_scratch_df = df.loc[df['Notes'] == 'e2eQPAT_from_scratch']
#print(f'Experimental from scratch shape: {experimental_from_scratch_df.shape}')
print_double_tex_reslts_table(
    get_metrics_dict(experimental_fine_tune_df, convert_m_to_cm=False),
    '{Fine-tune on experimental dataset}',
    get_metrics_dict(experimental_from_scratch_df, convert_m_to_cm=False),
    '{Train from scratch on experimental dataset}',
    '{Performance metrics for training on the experimental dataset. Mean and standard deviation of 5 runs.}',
    '{tab:experimental_test_metrics}'
)


print('============================== No Fluence, No lr Scheduler ==============================')
'''
# print TeX results table for pretraining on synthetic data
pretrain_df = df.loc[df['Notes'] == 'pretrain_no_fluence_no_lr_scheduler']
#print(f'Pretrain shape: {pretrain_df.shape}')
print_single_tex_reslts_table(
    get_metrics_dict(pretrain_df, convert_m_to_cm=True),
    '{Performance metrics for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs.}',
    '{tab:ImageNet_pretrain_metrics}'
)
'''