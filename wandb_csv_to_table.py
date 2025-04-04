import numpy as np
import pandas as pd
import scipy.stats as st

file = 'wandb_export_2025-03-18T12_28_37.136+00_00.csv'

models = ['UNet_e2eQPAT',
          'UNet_diffusion_ablation',
          'UNet_wl_pos_emb',
          'UNet_smp',
          'DDIM']

metrics_dict = {'mean_RMSE' : [None]*5, 'mean_MAE' : [None]*5, 'mean_Rel_Err' : [None]*5,
           'mean_R2' : [None]*5, 'mean_PSNR' : [None]*5, 'mean_SSIM' : [None]*5}
models_dict = {model : metrics_dict for model in models}
folds_str = np.array(['_0', '_1', '_2', '_3', '_4'])

# load dataframe and convert to numpy array
df = pd.read_csv(file)
columns = df.columns.values
pretrain = df.loc[df['Notes'] == 'pretrain']
digimouse = df.loc[df['Notes'] == 'digimouse']
digimouse_extrusion = df.loc[df['Notes'] == 'digimouse_extrusion']
experimental = df.loc[df['Notes'] == 'experimental']

for model in models:
    model_folds = np.char.add(np.array([model]), folds_str)
    # get each fold this model was pretrained using the synthetic data
    model_pretrain = pd.concat([
        pretrain.loc[pretrain['Name'] == model+'_0'],
        pretrain.loc[pretrain['Name'] == model+'_1'],
        pretrain.loc[pretrain['Name'] == model+'_2'],
        pretrain.loc[pretrain['Name'] == model+'_3'],
        pretrain.loc[pretrain['Name'] == model+'_4']
    ])
    for metric in metrics_dict.keys():
        metric_values = model_pretrain[metric].values
        # calculate mean and std for each metric
        models_dict[model][metric] = [np.mean(metric_values), np.std(metric_values)]   
 
# print TeX results table for preatraining on synthetic data
print(f'\begin{{table*}}                                                                                              \
\centering                                                                                                           \
\caption{{Performance for training on the synthetic ImageNet phantom dataset. Mean and standard deviation of 5 runs. }} \
\label{{table}}                                                                                                        \
\setlength{{\tabcolsep}}{{3pt}}                                                                                   \
%\begin{{tabular}}{{|p{{25pt}}|p{{75pt}}|p{{115pt}}|}}                                                \
\begin{{tabular}}{{|l|l|l|l|l|l|l|}}    \
\hline  \
Model & RMSE (cm$^{{-1}}$) & Abs. Error (cm$^{{-1}}$) & Rel. Error (\%) & R$^{{2}}$ & PSNR & SSIM \\    \
\hline  \
U-Net & & & & & & \\    \
Diffusion & & & & & & \\    \
\hline  \
\end{{tabular}}   \
\label{{tab:exp_phantom_metrics}}   \
\end{{table*}}  \
')