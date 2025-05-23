# Description: This script is used to scroll through the synthetic images for
# visual inspection. It is used to check the quality of the synthetic images
# It also written with the possibility of being adapted for data annotation

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from typing import Union
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utility_functions import square_centre_crop


class ImageViewer(QWidget):
    
    def __init__(self, sim_dirs : Union[set, tuple, list],
                 parent=None, 
                 load_all : bool=True, # load_all=True speeds up the application if all data fits into the memory, crashes if not
                 start_idx : int=0) -> None:
        super().__init__(parent)
        self.initUI()
        self.sim_dirs = list(sim_dirs)
        self.load_all = load_all
        self.image_names = {}
        self.data = {}
        print('gathering images...')
        for i, sim_dir in enumerate(sim_dirs):
            print(f'{i+1}/{len(sim_dirs)} {sim_dir}')
            [data, self.cfg] = load_sim(sim_dir, args='all', verbose=True, delete_incomplete_samples=False)
            keys = list(data.keys())
            if len(keys) != 0:
                if self.load_all:
                    self.data.update(data) # add images to data dict
                for image_name in keys:
                    self.image_names[image_name] = sim_dir
            else:
                print(f'no data found in {sim_dir}')
        # sort image names (alphabetically)
        self.image_names = dict(sorted(self.image_names.items()))
        print(f'{len(list(self.image_names.keys()))} images found')
        self.idx = start_idx # start at image
        self.sim_dir = list(self.image_names.items())[self.idx][1]
        if not self.load_all:
            print(f'loading {self.sim_dir}')
            [self.data, self.cfg] = load_sim(self.sim_dir, args='all', verbose=True)
        print(f'self.data.keys() {sorted(self.data.keys())}')
        self.labels = [r'Absorption $\mu_{a}$ (m$^{-1}$)',
                       r'Scattering $\mu_{s}$ (m$^{-1}$)',
                       r'Fluence $\Phi$ (J m$^{-2}$)', 
                       r'Initial pressure $p_{0}$ (Pa)',
                       r'Reconstrction $\hat{p}_{0}$ (Pa)',
                       'Mask']
        self.plot()
    
    def initUI(self):
        self.setGeometry(100, 100, 1600, 1600)
        self.setWindowTitle('Image Viewer (press left/right arrow key to scroll)')
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left and self.idx > 0:
            self.idx -= 1
            self.plot()
        elif event.key() == Qt.Key_Right and self.idx < len(self.image_names) - 1:
            self.idx += 1
            self.plot()

    def closeEvent(self, event) -> None:
        QApplication.quit()
        super().closeEvent(event)

    def plot(self):
        (image_name, sim_dir) = list(self.image_names.items())[self.idx]
        if sim_dir != self.sim_dir and not self.load_all:
            self.sim_dir = sim_dir
            print(f'loading {self.sim_dir}')
            [self.data, self.cfg] = load_sim(self.sim_dir, args='all', verbose=True)
            print(f'data.keys() {data.keys()}')
        images = []
        try:        
            images = np.asarray([self.data[image_name]['mu_a'],
                                 self.data[image_name]['mu_s'], 
                                 self.data[image_name]['Phi'],
                                 self.data[image_name]['mu_a']*self.data[image_name]['Phi'],
                                 self.data[image_name]['p0_tr'],
                                 square_centre_crop(self.data[image_name]['bg_mask'], self.cfg['crop_size'])])
            title = f'{image_name} in {sim_dir}'
            self.canvas.figure.clear()
            self.ImageViewerHeatmap(
                images, dx=self.cfg['dx'], rowmax=3, labels=self.labels, title=title
            )
            self.canvas.draw()
        except Exception as e:
            print(f'failed to load {image_name} from {sim_dir}\n{e}')
        
    def ImageViewerHeatmap(self, img : np.ndarray, 
                           title='', 
                           cmap='binary_r', 
                           vmax=None,
                           vmin=None,
                           dx=0.0001, 
                           rowmax=6,
                           labels=None,
                           sharescale=False,
                           cbar_label=None) -> None:
        # TODO: heatmap should use a list to plot images of different resolution
        # use cmap = 'cool' for feature extraction
        # use cmap = 'binary_r' for raw data
        dx = dx * 1e3 # [m] -> [mm]
        
        frames = []
        shape = np.shape(img)
        if sharescale or len(shape) == 2:
            mask = np.logical_not(np.isnan(img))
            if not vmin:
                vmin = np.min(img[mask])
            if not vmax:
                vmax = np.max(img[mask])
        
        extent = [-dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2]
        
        nframes = shape[0]
        nrows = int(np.ceil(nframes/rowmax))
        rowmax = nframes if nframes < rowmax else rowmax
        for row in range(nrows):
            for col in range(rowmax):
                idx = row * rowmax + col
                if idx >= nframes:
                    break
                ax = self.ax.figure.add_subplot(nrows, rowmax, idx + 1)
                if not sharescale:
                    mask = np.logical_not(np.isnan(img[idx]))
                    vmin = np.min(img[idx][mask])
                    vmax = np.max(img[idx][mask])
                frames.append(ax.imshow(
                    img[idx],
                    cmap=cmap, 
                    vmin=vmin, 
                    vmax=vmax,
                    extent=extent,
                    origin='lower'
                ))
                ax.set_xlabel('x (mm)')
                if labels:
                    ax.set(title=labels[idx])
                elif nframes > 1:
                    ax.set(title='pulse '+str(idx))
                if not sharescale:
                    divider = make_axes_locatable(ax)
                    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = self.ax.figure.colorbar(frames[idx], cax=cbar_ax, orientation='vertical')
                    if cbar_label:
                        cbar.set_label=cbar_label
        
        self.ax.figure.subplots_adjust(right=0.8)
        
        if sharescale:
            cbar_ax = self.ax.figure.add_axes([0.85, 0.15, 0.02, 0.7])
            cbar = self.ax.figure.colorbar(frames[0], cax=cbar_ax)
            if cbar_label:
                cbar.set_label(cbar_label)
        else:
            self.ax.figure.tight_layout()
                
        self.ax.figure.suptitle(title, fontsize='xx-large')
        

if __name__ == '__main__':
    #root_dir = '/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/3d_digimouse' # from wsl
    root_dir = '/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/2d_extrusion_digimouse' # from wsl
    #root_dir = 'F:\\cluster_MSOT_simulations\\digimouse_fluence_correction\\3d_digimouse' # from windows
    #root_dir = 'E:\\ImageNet_MSOT_simulations'

    h5_dirs = glob.glob(os.path.join(root_dir, '**/*.h5'), recursive=True)
    json_dirs = glob.glob(os.path.join(root_dir, '**/*.json'), recursive=True)
        
    h5_dirs = {os.path.dirname(file) for file in h5_dirs}
    json_dirs = {os.path.dirname(file) for file in json_dirs}
        
    sim_dirs = h5_dirs.intersection(json_dirs)
    print(f'Found {len(sim_dirs)} simulations')    
    
    app = QApplication([])
    viewer = ImageViewer(list(sim_dirs), load_all=True, start_idx=0)
    app.exec_()