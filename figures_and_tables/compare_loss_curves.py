import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



file = '/mnt/c/Users/wv00017/OneDrive - University of Surrey/Documents/Meetings and Submissions/IEEE NSS MIC 2025  transfer learning for fluence correction/wandb_export_2025-05-05T16_41_01.910+01_00.csv'
df = pd.read_csv(file)
columns = df.columns.values

print(columns)
print(df.head())

epochs = 200
steps = df['Step'].values
steps = steps * epochs / np.max(steps)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.set_yscale('log')
ax.set_ylabel(r'Mean Squared Error (a.u.)')
#ax.set_xlabel('Steps')
ax.set_xlabel('Epochs')

for fold in ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']:
    for training_type in ['from_scratch', 'fine_tune']:
        
        train_loss = df[f'UNet_e2eQPAT_{fold}_{training_type} - mean_train_loss'].values
        mask = np.isfinite(train_loss)
        train_loss = train_loss[mask]
        train_loss_steps = steps[mask]
        
        val_loss = df[f'UNet_e2eQPAT_{fold}_{training_type} - mean_experimental_val_loss'].values
        mask = np.isfinite(val_loss)
        val_loss = val_loss[mask]
        val_loss_steps = steps[mask]
        
        color = 'blue' if training_type == 'from_scratch' else 'red'
        if fold == 'fold0':
            if training_type == 'from_scratch':
                mean_train_loss_from_scratch = np.zeros_like(train_loss_steps, dtype=np.float32)
                mean_val_loss_from_scratch = np.zeros_like(val_loss_steps, dtype=np.float32)
                mean_train_loss_fine_tune = np.zeros_like(train_loss_steps, dtype=np.float32)
                mean_val_loss_fine_tune = np.zeros_like(val_loss_steps, dtype=np.float32)
                
            training_type = 'not pre-trained' if training_type == 'from_scratch' else 'pre-trained'
            ax.plot(train_loss_steps, train_loss, label=f'{training_type}, train loss', alpha=0.5, color=color, linestyle='-', linewidth=1)
            ax.plot(val_loss_steps, val_loss, label=f'{training_type}, validation loss', alpha=0.5, color=color, linestyle='--', linewidth=1)
        ax.plot(train_loss_steps, train_loss, alpha=0.5, color=color, linestyle='-', linewidth=1)
        ax.plot(val_loss_steps, val_loss, alpha=0.5, color=color, linestyle='--', linewidth=1)
        
        if training_type == 'from_scratch' or training_type == 'not pre-trained':
            mean_train_loss_from_scratch += train_loss / 5
            mean_val_loss_from_scratch += val_loss / 5
        else:
            mean_train_loss_fine_tune += train_loss / 5
            mean_val_loss_fine_tune += val_loss / 5
    
ax.plot(train_loss_steps, mean_train_loss_from_scratch, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
ax.plot(val_loss_steps, mean_val_loss_from_scratch, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
ax.plot(train_loss_steps, mean_train_loss_fine_tune, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
ax.plot(val_loss_steps, mean_val_loss_fine_tune, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    
# simple
#ax.plot(train_loss_steps, mean_train_loss_from_scratch, color='blue', linestyle='-', linewidth=1.5, label='not pre-trained, train loss')
#ax.plot(val_loss_steps, mean_val_loss_from_scratch, color='blue', linestyle='--', linewidth=1.5, label='not pre-trained, validation loss')
#ax.plot(train_loss_steps, mean_train_loss_fine_tune, color='red', linestyle='-', linewidth=1.5, label='pre-trained, train loss')
#ax.plot(val_loss_steps, mean_val_loss_fine_tune, color='red', linestyle='--', linewidth=1.5, label='pre-trained, validation loss')
    
ax.legend(loc='upper right')
ax.set_xlim(0, max(steps))
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()

fig.savefig('loss_curves.png')
fig.savefig('loss_curves.svg')
#fig.savefig('loss_curves_simple.png')
#fig.savefig('loss_curves_simple.svg')