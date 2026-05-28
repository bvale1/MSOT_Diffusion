import subprocess
import os
import itertools
import textwrap
from datetime import datetime

root_dir = 'preprocessing/20240517_BphP_cylinders_no_noise/'

#MODELS = ['UNet-e2eQPAT','EDM2','UNet_diffusion_ablation']
MODELS = ['UNet-e2eQPAT']

#FOLDS = [0, 1, 2, 3, 4]
#FOLDS = [1, 2, 3, 4]
FOLDS = [0]

DATASETS = {
    "ImageNet" : "/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/20260522_ImageNet_MSOT_Dataset",
    "Digimouse" : "/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/20260522_digimouse_MSOT_Dataset",
    "Digimouse_extrusion" : "/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/20260522_digimouse_extrusion_MSOT_Dataset",
    "e2eQPAT" : "/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/e2eQPAT_dataset",
}

EXPERIMENTS = [
    "ImageNet_pretrain",
    "Digimouse_test",
    "Digimouse_extrusion_test",
    "experimental_from_scratch",
    "experimental_fine_tune",
]
experiment = "ImageNet_pretrain"
#experiment = "Digimouse_test"
#experiment = "Digimouse_extrusion_test"
#experiment = "experimental_from_scratch"
#experiment = "experimental_fine_tune"

save_dirs = {
    'EDM2': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_EDM2",
    'UNet-e2eQPAT': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_UNet_e2eQPAT",
    'UNet_wl_pos_emb': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_UNet_wl_pos_emb",
    'UNet_diffusion_ablation': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_UNet_diffusion_ablation"
}

for model, fold in itertools.product(MODELS, FOLDS):
    memory = '32G' if model == 'UNet-e2eQPAT' else '64G'
    partition = '3090' if model == 'UNet-e2eQPAT' else 'a100'
    time_limit = '00-06:00:00' if model == 'UNet-e2eQPAT' else '00-72:00:00'
    synthetic_or_experimental = 'experimental' if experiment in ['experimental_from_scratch', 'experimental_fine_tune'] else 'synthetic'
    synthetic_dataset = 'Digimouse' if experiment in ['Digimouse_test'] else ('Digimouse_extrusion' if experiment in ['Digimouse_extrusion_test'] else 'ImageNet')
    epochs = '1000' if model == 'EDM2' else '200'
    wandb_notes = f"{experiment}_{model}_fold{fold}"

    sub_file = textwrap.dedent(f"""
    #!/bin/bash
                               
    ### Job Name ###
    #SBATCH --job-name={model}_fold{fold}

    ## CPU core requirements ###
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=4
    #SBATCH --ntasks-per-node=1

    ### CPU Memory (RAM) requirements ###
    #SBATCH --mem={memory}

    ### GPU requirements ###
    #SBATCH --partition={partition}
    #SBATCH --gpus=1

    ### Max. time requirement - DD-HH:MM:SS ###
    #SBATCH --time={time_limit}

    ### Job log files ###
    #SBATCH -o slurm.%j.%N.out
    #SBATCH -e slurm.%j.%N.err

    ### Apptainer execution ###
    apptainer exec oras://container-registry.surrey.ac.uk/shared-containers/billy-msot_diffusion-container:latest \
    python3 clone_and_run_msot_diffusion.py \
    --cluster_id .N$SLURM_JOB_NODELIST.j$SLURM_JOB_ID \
    --save_dir {save_dirs[model]} \
    --synthetic_or_experimental "{synthetic_or_experimental}" \
    --synthetic_root_dir {DATASETS[synthetic_dataset]} \
    --experimental_root_dir {DATASETS['e2eQPAT']} \
    --epochs {epochs} \
    --model {model} \
    --data_normalisation "standard" \
    --fold {fold} \
    --wandb_notes {wandb_notes}
    EOF
    """).strip() + "\n"

    submit_script_path = 'submit_MSOT_Diffusion.sh'
    with open(submit_script_path, 'w', encoding='utf-8') as f:
        f.write(sub_file)
    os.chmod(submit_script_path, 0o755)
    subprocess.run(['sbatch', submit_script_path], check=True)
    