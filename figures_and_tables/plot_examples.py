import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from compute_outputs import compute_outputs


models_dirs = [
    ('Not pre-trained', '20250430_UNet_e2eQPAT.Naisurrey26.j783005/RegressionUNet_epoch199.pt'),
    ('pre-trained', '20250501_UNet_e2eQPAT.Naisurrey25.j783389/RegressionUNet_epoch199.pt')    
]
colors = ['blue', 'red']
sample_indices = np.arange(10, 379, 21).astype(int)

outputs_dict, dx = compute_outputs(models_dirs, sample_indices)

plt.rcParams.update({'font.size': 12})
for i in range(len(sample_indices)):
    X = outputs_dict[models_dirs[0][0]]['X'][i]
    mu_a = outputs_dict[models_dirs[0][0]]['mu_a'][i]
    mu_a_hat = []
    for model, _ in models_dirs:
        mu_a_hat.append(outputs_dict[model]['mu_a_hat'][i])
    
    v_max_X = np.max(X)
    v_min_X = np.min(X)
    v_min_mu_a = min(np.min(mu_a), np.min(mu_a_hat))
    v_max_mu_a = max(np.max(mu_a), np.max(mu_a_hat))
    extent = [-dx*X.shape[-2]/2, dx*X.shape[-2]/2,
              -dx*X.shape[-1]/2, dx*X.shape[-1]/2]
    line_profile_ax = np.linspace(-dx*X.shape[-2]/2, dx*X.shape[-2]/2, X.shape[-2])
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    
    img1 = ax[0, 0].imshow(
        X, cmap='binary_r', vmin=v_min_X, vmax=v_max_X,
        origin='lower', extent=extent
    )
    #ax[0, 0].set_title('Reconstructed light energy deposition')
    divider = make_axes_locatable(ax[0, 0])
    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(img1, cax=cbar_ax, orientation='vertical')
    cbar1.set_label('a.u.')#(r'Pa J$^{-1}$')   
    ax[0, 0].set_xlabel('x (mm)')
    ax[0, 0].set_ylabel('z (mm)')
    
    img2 = ax[0, 1].imshow(
        mu_a - mu_a_hat[1], cmap='RdBu', vmin=-np.max(np.abs(mu_a - mu_a_hat[1])), 
        vmax=np.max(np.abs(mu_a - mu_a_hat[1])), origin='lower', extent=extent
    )
    #ax[0, 0].set_title(r'$\mu_{\text{a}} - \hat{\mu_{\text{a}}}$')
    divider = make_axes_locatable(ax[0, 1])
    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(img2, cax=cbar_ax, orientation='vertical')
    cbar2.set_label(r'cm$^{-1}$')   
    ax[0, 1].set_xlabel('x (mm)')
    ax[0, 1].set_ylabel('z (mm)')  
    
    img3 = ax[1, 0].imshow(
        mu_a, cmap='binary_r', vmin=v_min_mu_a, vmax=v_max_mu_a,
        origin='lower', extent=extent
    )
    ax[1, 0].plot(
        line_profile_ax, np.zeros_like(line_profile_ax, dtype=np.float32)-20*dx,
        color='red', linestyle='--'
    )
    #ax[1, 0].set_title('Absorption coefficient')
    divider = make_axes_locatable(ax[1, 0])
    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(img3, cax=cbar_ax, orientation='vertical')
    cbar3.set_label(r'cm$^{-1}$')   
    ax[1, 0].set_xlabel('x (mm)')
    ax[1, 0].set_ylabel('z (mm)')
    
    img4 = ax[1, 1].imshow(
        mu_a_hat[1], cmap='binary_r', vmin=v_min_mu_a, vmax=v_max_mu_a,
        origin='lower', extent=extent
    )
    ax[1, 1].plot(
        line_profile_ax, np.zeros_like(line_profile_ax, dtype=np.float32)-20*dx,
        color='red', linestyle='--'
    )
    #ax[1, 1].set_title('Predicted absorption coefficient')
    divider = make_axes_locatable(ax[1, 1])
    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    cbar4 = fig.colorbar(img4, cax=cbar_ax, orientation='vertical')
    cbar4.set_label(r'cm$^{-1}$')   
    ax[1, 1].set_xlabel('x (mm)')
    ax[1, 1].set_ylabel('z (mm)')
    
    fig.tight_layout()
    fig.savefig(f'example_{i}.png')
    
    # Plotting line profile
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(
        line_profile_ax, mu_a[(mu_a.shape[0]//2)-20,:],
        label='ground truth', color='black'
    )
    for j, (model_name, _) in enumerate(models_dirs):
        ax.plot(
            line_profile_ax, mu_a_hat[j][(mu_a.shape[0]//2)-20,:],
            label=model_name, linestyle='--', color=colors[j]
        )
    #ax.set_title('Line profile')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Absorption coefficient (cm$^{-1}$)')
    ax.set_xlim(-dx*X.shape[-2]/2, dx*X.shape[-2]/2)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01))
    fig.tight_layout()
    fig.savefig(f'line_profile_{i}.png')
    
    plt.close('all')