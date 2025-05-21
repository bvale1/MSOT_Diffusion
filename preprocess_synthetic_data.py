import numpy as np
import h5py, json, os, logging, glob, argparse
from dataloader import load_sim, delete_group_from_h5
from utility_functions import square_centre_crop

# This is a script to get the min, max, mean, and std of an entire dataset
# (normalisation parameters) and pack it into a h5 and json file
# the h5 files is structured as follows:
# dataset.h5
# ├── samples 
# │   ├── sample_0_name
# │   │   ├── X (reconstucted pressure image in Pa J^-1)
# │   │   ├── Phi (fluence image in m^-2)
# │   │   ├── corrected_image (corrected pressure image in m^-1 J^-1)
# │   │   ├── mu_a (absorption coefficient image in m^-1)
# │   │   ├── bg_mask (background mask)
# │   │   ├── wavelength_nm (wavelength in nm)
# │   │   ├── laser_energy (laser energy in J)
# │   ├── ...
# │   ├── sample_n_name
# │   │   ├── X (reconstucted pressure image in Pa J^-1)
# │   │   ├── Phi (fluence image in m^-2)
# │   │   ├── corrected_image (corrected pressure image in m^-1 J^-1)
# │   │   ├── mu_a (absorption coefficient image in m^-1)
# │   │   ├── bg_mask (background mask)
# │   │   ├── wavelength_nm (wavelength in nm)
# │   │   ├── laser_energy (laser energy in J)
# ├── train
# │   ├── 0 (fold 0 sample names)
# │   ├── ...
# │   ├── n (fold n sample names)
# ├── val
# │   ├── 0 (fold 0 sample names)
# │   ├── ...
# │   ├── n (fold n sample names)
# ├── test
# │   ├── 0 (fold 0 sample names)
# │   ├── ...
# │   ├── n (fold n sample names)

# corrected_image is depricated because the calculation numerically unstable
# use either absorption coefficient, or fluence as the target

# IMPORTANT:
# - If any samples do not pass the sanity checks and are skipped, they should be
#   deleted using the --delete_failed_samples flag, then this script should be 
#   run again to ensure the normalisation parameters are correct, and each
#   subset is the correct size
# - When testing on a model trained on a different dataset, ensure the 
#   normalisation parameters are the same as the training dataset, meaning
#   one should copy and paste the normalisation parameters from the training
#   dataset to the testing dataset

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
        default='20250327_ImageNet_MSOT_Dataset'
        #default='20250327_digimouse_MSOT_Dataset'
        #default='20250327_digimouse_extrusion_MSOT_Dataset'
    )
    parser.add_argument('--root_dir', type=str,
        default = '/mnt/e/ImageNet_MSOT_simulations' # from wsl
        #default = '/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/3d_digimouse'
        #default = '/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/2d_extrusion_digimouse'
    )
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--git_hash', type=str, default=None)
    parser.add_argument(
        '--delete_failed_samples', action='store_true', default=False, 
        help='delete samples that do not pass the vibe check (sanity checks)'
    )
    
    args = parser.parse_args()
    
    dataset_cfg = {
        'dataset_name': args.dataset_name,
        'root_dir': args.root_dir,
        'git_hash': args.git_hash,
        'n_images': 0,
        'dx' : 0.0001,
        'crop_size' : 256,
        'train_val_test_split' : [0.8, 0.1, 0.1], # 80% train, 10% val, 10% test
        #'train_val_test_split' : [0.0, 0.0, 1.0], # use all data for testing
        'folds' : 5,
        #'folds' : 1,
        'units' : {
            'X' : 'Pa J^-1', 'Phi' : 'm^-2', 
            #'corrected_image' : 'm^-1 J^-1',
            'mu_a' : 'm^-1'
        },
        'normalisation_X': {
            'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0
        },
        'normalisation_Phi': {
            'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0
        },
        #'normalisation_corrected_image': {
        #    'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0
        #},
        'normalisation_mu_a': {
            'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0
        }
    }
    if (dataset_cfg['train_val_test_split'][1]+dataset_cfg['train_val_test_split'][2]) * dataset_cfg['folds'] > 1.0:
        raise ValueError(f'Not enough samples for {dataset_cfg['folds']} folds with train_test_split={dataset_cfg['train_val_test_split']}')
    
    file_path = os.path.join(args.output_dir, dataset_cfg['dataset_name'], 'dataset.h5')
    if os.path.exists(file_path):
        logging.info(f"{file_path} already exists and will be overwritten")
    else:
        os.makedirs(os.path.join(args.output_dir, dataset_cfg['dataset_name']))
    with h5py.File(file_path, 'w') as f:
        logging.info(f"creating {file_path}")
        f.create_group('samples')
        for subset in ['train', 'val', 'test']:
            f.create_group(subset)
            for fold in range(dataset_cfg['folds']):
                f[subset].create_group(str(fold))        
    
    h5_dirs = glob.glob(os.path.join(args.root_dir, '**/*.h5'), recursive=True)
    json_dirs = glob.glob(os.path.join(args.root_dir, '**/*.json'), recursive=True)
    
    h5_dirs = {os.path.dirname(file) for file in h5_dirs}
    json_dirs = {os.path.dirname(file) for file in json_dirs}
    
    sim_dirs = h5_dirs.intersection(json_dirs)
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Found {len(h5_dirs)} h5 files')

    n_images = 0
    sample_names = []
    non_empty_sim_dirs = []
    for i, sim in enumerate(sim_dirs):
        with h5py.File(os.path.join(sim, 'data.h5'), 'r') as f:
            logging.info(f'Found {len(list(f.keys()))} images in {sim} ({i+1}/{len(sim_dirs)})')
            if len(list(f.keys())) != 0:
                n_images += len(list(f.keys()))
                sample_names += list(f.keys())
                non_empty_sim_dirs.append(sim)

    np.random.seed(42)
    np.random.shuffle(sample_names)
    # normalise the train, val, and test splits
    train_val_test_split = np.asarray(dataset_cfg['train_val_test_split'])
    train_val_test_split /= np.sum(train_val_test_split)
    dataset_cfg['train_val_test_split'] = train_val_test_split.tolist()
    
    # split the samples into train, val, and test sets
    with h5py.File(file_path, 'r+') as f:
        for fold in range(dataset_cfg['folds']):
            val_start_idx = int(len(sample_names) * ((train_val_test_split[1]+train_val_test_split[2])*fold))
            val_end_idx = val_start_idx + int(len(sample_names)*train_val_test_split[1])
            test_start_idx = val_end_idx
            test_end_idx = val_end_idx + int(len(sample_names) * (train_val_test_split[2]))
            print(f'fold={fold}, n_samples={len(sample_names)}')
            print(f'val_start_idx={val_start_idx}, val_end_idx={val_end_idx}')
            print(f'test_start_idx={test_start_idx}, test_end_idx={test_end_idx}')
            f['train'][str(fold)].create_dataset(
                'sample_names',
                data=np.array(sample_names[:val_start_idx] + sample_names[test_end_idx:], dtype='S'),
                #dtype=h5py.string_dtype(encoding='utf-8')
            )
            f['val'][str(fold)].create_dataset(
                'sample_names',
                data=np.array(sample_names[val_start_idx:val_end_idx], dtype='S'),
                #dtype=h5py.string_dtype(encoding='utf-8')
            )
            f['test'][str(fold)].create_dataset(
                'sample_names', 
                data=np.array(sample_names[test_start_idx:test_end_idx], dtype='S'),
                #dtype=h5py.string_dtype(encoding='utf-8')
            )

    # rolling averages used to calculate normalisation parameters to avoid
    # floating point precision errors and conserve memory
    dataset_cfg['n_images'] = n_images
    
    for i, sim in enumerate(non_empty_sim_dirs):
        logging.info(f'Loading {sim} ({i+1}/{len(non_empty_sim_dirs)})')
        [data, cfg] = load_sim(sim)
        sim_name = sim.replace('\\', '/')
        
        if i == 0: # assume all simulations have the same dx and square crop size
            dataset_cfg['dx'] = cfg['dx']
            dataset_cfg['crop_size'] = cfg['crop_size']
        
        for j, image in enumerate(data.keys()):
            try:
                wavelength_nm = int(cfg['wavelengths'][0] * 1e9) # convert m to nm
            except:
                # wavelength is not in the config file of digimouse simulations
                wavelength_nm = image.split('__')[-1].split('_')[-1]
            
            group_name = (sim_name.split('/')[-1]+'_'+image.split('__')[-1]).replace('.','')
            
            if ('p0_tr' not in data[image].keys()):
                logging.info(f'p0_tr not found in {group_name}, skipping sample')
                if args.delete_failed_samples:
                    delete_group_from_h5(file_path=sim, group_name=image)
                continue
            if ('Phi' not in data[image].keys()):
                logging.info(f'Phi not found in {group_name}, skipping sample')
                if args.delete_failed_samples:
                    delete_group_from_h5(file_path=sim, group_name=image)
                continue
            if ('mu_a' not in data[image].keys()):
                logging.info(f'mu_a not found in {group_name}, skipping sample')
                if args.delete_failed_samples:
                    delete_group_from_h5(file_path=sim, group_name=image)
                continue
            if ('bg_mask' not in data[image].keys()):
                logging.info(f'bg_mask not found in {group_name}, skipping sample')
                if args.delete_failed_samples:
                    delete_group_from_h5(file_path=sim, group_name=image)
                continue
                
            # X is the reconstrcuted pressure image divided by the laser energy (laser energy normalisation)
            X = data[image]['p0_tr'] / cfg['LaserEnergy'][j]
            # Phi is the fluence and is also normalised by the laser energy before being saved
            Phi = data[image]['Phi'] / cfg['LaserEnergy'][j] # [J m^-2] -> [m^-2]
            # corrected_image is the fluence (Phi) corrected image
            # fluence (Phi) is clamped at 1e-8 to avoid division by zero
            #Phi_clamped = Phi.copy()
            #Phi_clamped[Phi_clamped < 1e-8] = 1e-8
            # due to the clamping, the corrected image is also clampe
            #corrected_image = data[image]['p0_tr'] / Phi_clamped # [Pa J^-1] / [m^-2] -> [J m^-3 J^-1] * [m^2] -> [m^-1]
            # corrected image is depricated because the calculation numerically unstable
            # either absorption coefficient, or fluence may be used as the target
            mu_a = data[image]['mu_a']            
            
            bg_mask = square_centre_crop(
                np.squeeze(data[image]['bg_mask']), dataset_cfg['crop_size']
            )
            
            if np.any(mu_a < 0): # sanity checking
                logging.info(f'{group_name} absorption coefficient less than zero, skipping sample')
                logging.info(f'np.min(mu_a)={np.min(mu_a)}, wavelength={cfg["wavelengths"]}')
                logging.info(f'{100*np.sum(mu_a < 0)/np.prod(mu_a.shape)}% less than zero')
                if args.delete_failed_samples:
                    delete_group_from_h5(file_path=sim, group_name=image)
                continue
            if np.any(mu_a < mu_a[0,0]): # sanity checking
                logging.info(f'{group_name} absorption coefficient less than coupling medium, skipping sample')
                logging.info(f'np.min(mu_a)={np.min(mu_a)}, np.min(mu_a[0,0])={np.min(mu_a[0,0])}, wavelength={cfg["wavelengths"]}')
                logging.info(f'{100*np.sum(mu_a < mu_a[0,0])/np.prod(mu_a.shape)}% less than zero')
                if args.delete_failed_samples:
                    delete_group_from_h5(file_path=sim, group_name=image)
                continue
                
            dataset_cfg['normalisation_X']['max'] = max(dataset_cfg['normalisation_X']['max'], float(np.max(X)))
            dataset_cfg['normalisation_X']['min'] = min(dataset_cfg['normalisation_X']['min'], float(np.min(X)))
            dataset_cfg['normalisation_X']['mean'] += float(np.mean(X) / n_images)
            dataset_cfg['normalisation_Phi']['max'] = max(dataset_cfg['normalisation_Phi']['max'], float(np.max(Phi)))
            dataset_cfg['normalisation_Phi']['min'] = min(dataset_cfg['normalisation_Phi']['min'], float(np.min(Phi)))
            dataset_cfg['normalisation_Phi']['mean'] += float(np.mean(Phi) / n_images)
            #dataset_cfg['normalisation_corrected_image']['max'] = max(dataset_cfg['normalisation_corrected_image']['max'], float(np.max(corrected_image)))
            #dataset_cfg['normalisation_corrected_image']['min'] = min(dataset_cfg['normalisation_corrected_image']['min'], float(np.min(corrected_image)))
            #dataset_cfg['normalisation_corrected_image']['mean'] += float(np.mean(corrected_image) / n_images)
            dataset_cfg['normalisation_mu_a']['max'] = max(dataset_cfg['normalisation_mu_a']['max'], float(np.max(mu_a)))
            dataset_cfg['normalisation_mu_a']['min'] = min(dataset_cfg['normalisation_mu_a']['min'], float(np.min(mu_a)))
            dataset_cfg['normalisation_mu_a']['mean'] += float(np.mean(mu_a) / n_images)
            
            with h5py.File(file_path, 'r+') as f:
                image_group = f['samples'].require_group(image)
                image_group.create_dataset('X', data=X, dtype=np.float32)
                image_group.create_dataset('Phi', data=Phi, dtype=np.float32)
                #image_group.create_dataset('corrected_image', data=corrected_image, dtype=np.float32)
                image_group.create_dataset('mu_a', data=mu_a, dtype=np.float32)
                image_group.create_dataset('bg_mask', data=bg_mask, dtype=bool)
                image_group.create_dataset('wavelength_nm', data=wavelength_nm, dtype=int)
                image_group.create_dataset('laser_energy', data=cfg['LaserEnergy'][j], dtype=float)
                
    ssr_X = 0.0
    ssr_phi = 0.0
    #ssr_corrected_image = 0.0
    ssr_mu_a = 0.0
    
    # calculate standard deviation
    logging.info(f'Calculating standard deviations')
    with h5py.File(file_path, 'r') as f:
        for i, image in enumerate(list(f['samples'].keys())):
            denomitator = np.prod(f['samples'][image]['X'].shape) * n_images - 1
            ssr_X += np.sum((f['samples'][image]['X'][()] - dataset_cfg['normalisation_X']['mean'])**2) / denomitator
            ssr_phi += np.sum((f['samples'][image]['Phi'][()] - dataset_cfg['normalisation_Phi']['mean'])**2) / denomitator
            #ssr_corrected_image += np.sum((f['samples'][image]['corrected_image'][()] - dataset_cfg['normalisation_corrected_image']['mean'])**2) / denomitator
            ssr_mu_a += np.sum((f['samples'][image]['mu_a'][()] - dataset_cfg['normalisation_mu_a']['mean'])**2) / denomitator
                
    dataset_cfg['normalisation_X']['std'] = float(np.sqrt(ssr_X))
    dataset_cfg['normalisation_Phi']['std'] = float(np.sqrt(ssr_phi))
    #dataset_cfg['normalisation_corrected_image']['std'] = float(np.sqrt(ssr_corrected_image))
    dataset_cfg['normalisation_mu_a']['std'] = float(np.sqrt(ssr_mu_a))
    
    print(f'dataset_cfg {dataset_cfg}')
    
    with open(os.path.join(args.output_dir, dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f, indent='\t')