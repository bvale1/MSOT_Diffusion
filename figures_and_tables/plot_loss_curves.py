import matplotlib.pyplot as plt
import numpy as np
from typing import Literal


def plot_loss_curves(
    model_dict : dict,
    run_notes : list, 
    epochs : int, 
    labels : list,
    savename : str,
    std_or_iqr : Literal['std', 'iqr'] = 'std'
    ) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_yscale('log')
    ax.set_ylabel(r'Mean Squared Error (a.u.)')
    #ax.set_xlabel('Steps')
    ax.set_xlabel('Epochs')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'teal']
    #lines = ['-', '--', '-.', ':']
    if __name__ == "__main__":
        for i, notes in enumerate(run_notes):
            notes_history = model_dict[notes]
            for j, fold in enumerate(['fold0', 'fold1', 'fold2', 'fold3', 'fold4']):
                for key in list(notes_history.keys()):
                    if fold in key:
                        notes_fold_history = notes_history[key]
                        break
                
                if fold == 'fold0':
                    val_loss_steps = notes_fold_history['mean_experimental_val_loss']['_step'].values
                    val_loss_steps = val_loss_steps * epochs / np.max(val_loss_steps)
                    try:
                        train_loss_key = 'mean_experimental_train_loss'
                        train_loss_steps =  notes_fold_history['mean_experimental_train_loss']['_step'].values
                    except:
                        train_loss_key = 'mean_train_loss'
                        train_loss_steps =  notes_fold_history['mean_train_loss']['_step'].values
                    train_loss_steps = train_loss_steps * epochs / np.max(train_loss_steps)
                
                    train_loss = np.full((5, len(train_loss_steps)), dtype=np.float32, fill_value=np.nan)
                    val_loss = np.full((5, len(val_loss_steps)), dtype=np.float32, fill_value=np.nan)

                # handles cases where some folds have different lengths of training
                fold_train_loss = notes_fold_history[train_loss_key][train_loss_key].values
                fold_val_loss = notes_fold_history['mean_experimental_val_loss']['mean_experimental_val_loss'].values
                train_loss[j, :len(fold_train_loss)] = fold_train_loss
                val_loss[j, :len(fold_val_loss)] = fold_val_loss
            
            if std_or_iqr == 'std':
                # plot mean train and val loss
                ax.plot(train_loss_steps, np.nanmean(train_loss, axis=0), color=colors[i], linestyle='-', label=f'{labels[i]} train loss')
                ax.plot(val_loss_steps, np.nanmean(val_loss, axis=0), color=colors[i], linestyle='--', label=f'{labels[i]}, validation loss')
                # fill between mean +/- std for train and val loss
                ax.fill_between(train_loss_steps, np.nanmean(train_loss, axis=0) - np.nanstd(train_loss, axis=0), np.nanmean(train_loss, axis=0) + np.nanstd(train_loss, axis=0), color=colors[i], alpha=0.2)
                ax.fill_between(val_loss_steps, np.nanmean(val_loss, axis=0) - np.nanstd(val_loss, axis=0), np.nanmean(val_loss, axis=0) + np.nanstd(val_loss, axis=0), color=colors[i], alpha=0.2)
            elif std_or_iqr == 'iqr':
                # plot median train and val loss
                ax.plot(train_loss_steps, np.nanmedian(train_loss, axis=0), color=colors[i], linestyle='-', label=f'{labels[i]} train loss')
                ax.plot(val_loss_steps, np.nanmedian(val_loss, axis=0), color=colors[i], linestyle='--', label=f'{labels[i]}, validation loss')
                # fill between 25th and 75th percentile for train and val loss
                ax.fill_between(train_loss_steps, np.nanpercentile(train_loss, 25, axis=0), np.nanpercentile(train_loss, 75, axis=0), color=colors[i], alpha=0.2)
                ax.fill_between(val_loss_steps, np.nanpercentile(val_loss, 25, axis=0), np.nanpercentile(val_loss, 75, axis=0), color=colors[i], alpha=0.2)
            
        ax.legend(loc='upper right')
        ax.set_xlim(0, epochs)
        #ax.set_ylim(1e-5, 10)
        ax.grid(True)
        ax.set_axisbelow(True)
        fig.tight_layout()

        fig.savefig(savename, dpi=300, bbox_inches='tight', format='pdf')


run = [
    #'e2eQPAT_Janeks_weights', # no loss curves for pretrained weights
    #'experimental_from_scratch_UNet_e2eQPAT',
    #'experimental_from_scratch_UNet_e2eQPAT_fixedseed',
    #'experimental_from_scratch_UNet_e2eQPAT_fixedseed_9bd6c0b',
    ###'e2eQPAT_from_scratch_amsgrad_lrem4_eps1em8', # <- final choice
    ###'e2eQPAT_fine_tune_and_pretained_with_amsgrad_lr1em4_eps1em8', # <- final choice
    #'e2eQPAT_fine_tune',
    #'e2eQPAT_fine_tune_no_shuffle',
    #'e2eQPAT_boft3_fine_tune',
    #'e2eQPAT_fine_tune_lr1em5_l2reg1em1',
    #'e2eQPAT_fine_tune_lr1em5',
    #'experimental_from_scratch_UNet_diffusion_ablation',
    #'experimental_from_scratch_UNet_diffusion_ablation_deeperunet',
    #'experimental_from_scratch_UNet_diffusion_ablation_ema',
    #'experimental_from_scratch_UNet_wl_pos_emb',
    'experimental_from_scratch_EDM2_lr1em3', # <- final choice
    #'experimental_from_scratch_EDM2_lr1em3_test',
    #'EDM2_fine_tune',
    #'EDM2_fine_tune_lr1em3',
    #'EDM2_fine_tune_lr1em3_test',
    #'EDM2_fine_tune2_lr1em3',
    #'EDM2_fine_tune_lr1em2_l2reg1em1',
    #'EDM2_fine_tune_lr1em3_l2reg1em1',
    'EDM2_fine_tune_lr1em3_l2reg1', # <- final choice
    #'EDM2_fine_tune_lr1em3_l2reg1_test',
    #'experimental_from_scratch_EDM2',
    
]

