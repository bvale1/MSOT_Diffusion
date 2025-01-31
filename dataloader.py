import numpy as np
import matplotlib.pyplot as plt
import h5py, json, logging, os
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union

def load_sim(path : str,
             args : Union[str, tuple, list]='all',
             verbose : bool=False, 
             delete_incomplete_samples : bool=False) -> list:
    data = {}
    with h5py.File(os.path.join(path, 'data.h5'), 'r+') as f:
        images = list(f.keys())
        if verbose:
            print(f'images found {images}')
        if args == 'all':
            args = f[images[0]].keys()
            if verbose:
                print(f'args found in images[0] {args}')
        for image in images:
            data[image] = {}
            for arg in args:
                if arg not in f[image].keys():
                    print(f'arg {arg} not found in {image}')
                    if delete_incomplete_samples:
                        try:
                            print(f'deleting {image} from {path}')
                            del f[image]
                            del data[image]
                        except Exception as e:
                            print(f'could not delete {image} {e}')
                    continue
                # include 90 deg anticlockwise rotation
                elif arg != 'sensor_data':
                    data[image][arg] = np.rot90(
                        np.array(f[image][arg][()]), k=1, axes=(-2,-1)
                    ).copy()
                else:
                    data[image][arg] = np.array(f[image][arg][()])
            
    with open(path+'/config.json', 'r') as f:
        cfg = json.load(f)
        
    return [data, cfg]


def delete_group_from_h5(file_path, group_name):
    file_path = os.path.join(file_path, 'data.h5')
    with h5py.File(file_path, 'r+') as f:
        if group_name in f:
            del f[group_name]
            

def heatmap(img, 
            title='', 
            cmap='binary_r', 
            vmax=None,
            vmin=None,
            dx=0.0001, 
            rowmax=6,
            labels=None,
            sharescale=False,
            cbar_label=None):
    # TODO: heatmap should use a list to plot images of different resolution
    logging.basicConfig(level=logging.INFO)    
    # use cmap = 'cool' for feature extraction
    # use cmap = 'binary_r' for raw data
    dx = dx * 1e3 # [m] -> [mm]
    
    frames = []
    
    # convert to numpy for plotting
    if type(img) == torch.Tensor:
        img = img.detach().numpy()
        
    shape = np.shape(img)
    if sharescale or len(shape) == 2:
        mask = np.logical_not(np.isnan(img))
        if not vmin:
            vmin = np.min(img[mask])
        if not vmax:
            vmax = np.max(img[mask])
    
    extent = [-dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2]
    
    if len(shape) == 2: # one pulse
        nframes = 1
        fig, ax = plt.subplots(nrows=1, ncols=nframes, figsize=(6,8))
        ax = np.array([ax])
        ax[0].set_xlabel('x (mm)')
        ax[0].set_ylabel('z (mm)')
        frames.append(ax[0].imshow(
            img,
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax,
            extent=extent,
            origin='lower'
        ))
        divider = make_axes_locatable(ax[0])
        cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(frames[0], cax=cbar_ax, orientation='vertical')    
        
    else: # multiple pulses
        nframes = shape[0]
        nrows = int(np.ceil(nframes/rowmax))
        rowmax = nframes if nframes < rowmax else rowmax
        fig, ax = plt.subplots(nrows=nrows, ncols=rowmax, figsize=(16, 12))
        ax = np.asarray(ax)
        if len(np.shape(ax)) == 1:
            ax = ax.reshape(1, rowmax)
        for row in range(nrows):
            ax[row, 0].set_ylabel('z (mm)')
        for col in range(rowmax):
            ax[-1, col].set_xlabel('x (mm)')
        ax = ax.ravel()
        
        for frame in range(nframes): 
            if not sharescale:
                mask = np.logical_not(np.isnan(img[frame]))
                vmin = np.min(img[frame][mask])
                vmax = np.max(img[frame][mask])
            frames.append(ax[frame].imshow(
                img[frame],
                cmap=cmap, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower'
            ))
            ax[frame].set_xlabel('x (mm)')
            if labels:
                ax[frame].set(title=labels[frame])
            elif nframes > 1:
                ax[frame].set(title='pulse '+str(frame))
            if not sharescale:
                divider = make_axes_locatable(ax[frame])
                cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(frames[frame], cax=cbar_ax, orientation='vertical')
                #cbar = plt.colorbar(frames[frame], ax=ax[frame])
                if cbar_label:
                    cbar.set_label=cbar_label

    fig.subplots_adjust(right=0.8)
    
    if sharescale:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(frames[0], cax=cbar_ax)
        if cbar_label:
            cbar.set_label=cbar_label
    else:
        fig.tight_layout()
            
    fig.suptitle(title, fontsize='xx-large')
    
    return (fig, ax, frames)



if __name__ == '__main__':
    # script to load and visualise a dataset
    #path = '\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\python_BphP_MSOT_sim\\test_runs\\unnamed_sim'
    path= 'F:\\cluster_MSOT_simulations\\digimouse_fluence_correction\\3d_digimouse\\20240906_digimouse_phantom.c187370.p0'
    
    logging.basicConfig(level=logging.INFO)
    
    [data, cfg] = load_sim(path, verbose=True)
    groups = list(data.keys())
    labels = [r'$\mu_{a}$ (m$^{-1}$)', r'$\mu_{s}$ (m$^{-1}$)',
              r'$\Phi$ (J m$^{-2}$)', r'$p_{0}$ initial pressure (Pa)',
              r'$\hat{p}_{0}$ time reversal (Pa)', r'$\mid{\hat{p}_{0}-p_{0}}\mid$ (Pa)',
              r'$\hat{p}_{0}/\Phi$ (m$^{-1}$)', r'$\mid{\mu_{a}-\hat{p}_{0}/\Phi}\mid$ (m$^{-1}$)']
    for i in range(min(len(groups), len(groups))):
        #images = [data[groups[i]]['mu_a'], data[groups[i]]['mu_s']]
        print(groups[i])
        images = [data[groups[i]]['mu_a'], data[groups[i]]['mu_s'], 
                  data[groups[i]]['Phi'], data[groups[i]]['mu_a']*data[groups[i]]['Phi'],
                  data[groups[i]]['p0_tr'], np.abs((data[groups[i]]['mu_a']*data[groups[i]]['Phi'])-data[groups[i]]['p0_tr']),
                  data[groups[i]]['p0_tr']/(data[groups[i]]['Phi']+1e-8),
                  np.abs(data[groups[i]]['mu_a']-(data[groups[i]]['p0_tr']/(data[groups[i]]['Phi']+1e-8)))]
        heatmap(images, dx=cfg['dx'], rowmax=4, labels=labels)