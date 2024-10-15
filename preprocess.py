import numpy as np
import h5py, json, os, logging, glob, argparse
from dataloader import load_sim

# This is a script to get the min, max, mean, and std of an entire dataset
# (normalisation parameters) and pack it into a h5 and json file

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ImageNet_MSOT_Dataset')
    parser.add_argument('--root_dir', type=str,
        #default = '/mnt/f/cluster_MSOT_simulations/ImageNet_fluence_correction' # from wsl
        #default = 'F:\\cluster_MSOT_simulations\\ImageNet_fluence_correction' # from windows
        default= 'E:/ImageNet_MSOT_simulations'
    )
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--git_hash', type=str, default=None)
    
    args = parser.parse_args()
    
    dataset_cfg = {
        'dataset_name': args.dataset_name,
        'root_dir': args.root_dir,
        'git_hash': args.git_hash,
        'n_images': 0,
        'dx' : 0.0001,
        'units' : {'X' : 'Pa J^-1', 'Y' : 'm^-1', 'mu_a' : 'm^-1'},
        'normalisation_X': {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0
        },
        'normalisation_Y': {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0
        },
        'normalisation_mu_a': {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0
        }
    }
    
    file_path = os.path.join(args.output_dir, dataset_cfg['dataset_name'], 'dataset.h5')
    if os.path.exists(file_path):
        logging.info(f"{file_path} already exists and will be overwritten")
    else:
        os.makedirs(os.path.join(args.output_dir, dataset_cfg['dataset_name']))
    with h5py.File(file_path, 'w') as f:
        logging.info(f"creating {file_path}")
        
    
    h5_dirs = glob.glob(os.path.join(args.root_dir, '**/*.h5'), recursive=True)
    json_dirs = glob.glob(os.path.join(args.root_dir, '**/*.json'), recursive=True)
    
    h5_dirs = {os.path.dirname(file) for file in h5_dirs}
    json_dirs = {os.path.dirname(file) for file in json_dirs}
    
    sim_dirs = h5_dirs.intersection(json_dirs)
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Found {len(h5_dirs)} h5 files')
    
    n_images = 0
    non_empty_sim_dirs = []
    for i, sim in enumerate(sim_dirs):
        with h5py.File(os.path.join(sim, 'data.h5'), 'r') as f:
            n_images += len(list(f.keys()))
            logging.info(f'Found {len(list(f.keys()))} images in {sim} ({i+1}/{len(sim_dirs)})')
            if len(list(f.keys())) != 0:
                non_empty_sim_dirs.append(sim)
                
    
    # rolling averages used to calculate normalisation parameters to avoid
    # floating point precision errors and conserve memory
    dataset_cfg['n_images'] = n_images
    
    for i, sim in enumerate(non_empty_sim_dirs):
        logging.info(f'Loading {sim} ({i+1}/{len(non_empty_sim_dirs)})')
        [data, cfg] = load_sim(sim)
        
        if i == 0:
            dataset_cfg['dx'] = cfg['dx']
        
        for j, image in enumerate(data.keys()):
            group_name = (sim.split('\\')[-1]+'_'+image.split('__')[-1]).replace('.','')
            # X is the pressure image divided by the laser energy
            # (laser energy normalisation)
            
            if ('p0_tr' not in data[image].keys()):
                logging.info(f'p0_tr not found in {group_name}, skipping sample')
                continue
            if ('Phi' not in data[image].keys()):
                logging.info(f'Phi not found in {group_name}, skipping sample')
                continue
            if ('mu_a' not in data[image].keys()):
                logging.info(f'mu_a not found in {group_name}, skipping sample')
                continue
                
            X = data[image]['p0_tr'] / cfg['LaserEnergy'][j]
            # Y is the fluence (Phi) corrected image
            # fluence (Phi) is clamped at 1e-8 to avoid division by zero
            Phi = data[image]['Phi']
            Phi[Phi < 1e-8] = 1e-8
            Y = data[image]['p0_tr'] / Phi
            # absorption coefficient may also be used as a target instead of 
            # the fluence corrected image
            mu_a = data[image]['mu_a']
            
            if np.any(mu_a < 0): # sanity checking
                logging.info(f'{group_name} absorption coefficient less than zero, skipping sample')
                logging.info(f'np.min(mu_a)={np.min(mu_a)}, wavelength={cfg["wavelengths"]}')
                logging.info(f'{100*np.sum(mu_a < 0)/np.prod(mu_a.shape)}% less than zero')
                continue
                
            dataset_cfg['normalisation_X']['max'] = max(dataset_cfg['normalisation_X']['max'], float(np.max(X)))
            dataset_cfg['normalisation_X']['min'] = min(dataset_cfg['normalisation_X']['min'], float(np.min(X)))
            dataset_cfg['normalisation_X']['mean'] += float(np.mean(X) / n_images)
            dataset_cfg['normalisation_Y']['max'] = max(dataset_cfg['normalisation_Y']['max'], float(np.max(Y)))
            dataset_cfg['normalisation_Y']['min'] = min(dataset_cfg['normalisation_Y']['min'], float(np.min(Y)))
            dataset_cfg['normalisation_Y']['mean'] += float(np.mean(Y) / n_images)
            dataset_cfg['normalisation_mu_a']['max'] = max(dataset_cfg['normalisation_mu_a']['max'], float(np.max(mu_a)))
            dataset_cfg['normalisation_mu_a']['min'] = min(dataset_cfg['normalisation_mu_a']['min'], float(np.min(mu_a)))
            dataset_cfg['normalisation_mu_a']['mean'] += float(np.mean(mu_a) / n_images)
            
            
            with h5py.File(file_path, 'r+') as f:
                group = f.require_group(group_name)
                group.create_dataset('X', data=X, dtype=np.float32)
                group.create_dataset('Y', data=Y, dtype=np.float32)
                group.create_dataset('mu_a', data=mu_a, dtype=np.float32)
                
    ssr_X = 0.0
    ssr_Y = 0.0
    ssr_mu_a = 0.0
    
    # calculate standard deviation
    logging.info(f'Calculating standard deviations')
    with h5py.File(file_path, 'r') as f:
        for i, image in enumerate(list(f.keys())):
            denomitator = np.prod(f[image]['X'].shape) * n_images - 1
            ssr_X += np.sum((f[image]['X'][()] - dataset_cfg['normalisation_X']['mean'])**2) / denomitator
            ssr_Y += np.sum((f[image]['Y'][()] - dataset_cfg['normalisation_Y']['mean'])**2) / denomitator
            ssr_mu_a += np.sum((f[image]['mu_a'][()] - dataset_cfg['normalisation_mu_a']['mean'])**2) / denomitator
            if i % 100 == 0:
                logging.info(f'{image}, {i+1}/{len(list(f.keys()))}, ssr_X={ssr_X}, ssr_Y={ssr_Y}, ssr_mu_a={ssr_mu_a}')
                
    dataset_cfg['normalisation_X']['std'] = float(np.sqrt(ssr_X))
    dataset_cfg['normalisation_Y']['std'] = float(np.sqrt(ssr_Y))
    dataset_cfg['normalisation_mu_a']['std'] = float(np.sqrt(ssr_mu_a))
    
    print(f'dataset_cfg {dataset_cfg}')
    
    with open(os.path.join(args.output_dir, dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f, indent='\t')