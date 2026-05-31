import subprocess
import os
import itertools
import textwrap
from datetime import datetime

pretrained_edm2=(
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_edm2_models/20250731_EDM2.Naisurrey22.j2010491"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_edm2_models/20250731_EDM2.Naisurrey22.j2010492"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_edm2_models/20250731_EDM2.Naisurrey23.j2010493"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_edm2_models/20250731_EDM2.Naisurrey23.j2010494"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_edm2_models/20250731_EDM2.Naisurrey24.j2010490"
)
pretrained_unet_e2eqpat=(
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_unete2eqpat/20251124_UNet_e2eQPAT.Naisurrey25.j2017600"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_unete2eqpat/20251124_UNet_e2eQPAT.Naisurrey25.j2017612"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_unete2eqpat/20251124_UNet_e2eQPAT.Naisurrey26.j2017617"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_unete2eqpat/20251124_UNet_e2eQPAT.Naisurrey25.j2017653"
"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/pretrained_unete2eqpat/20251124_UNet_e2eQPAT.Naisurrey25.j2017599"
)

#MODELS = ['UNet_e2eQPAT','EDM2','UNet_diffusion_ablation']
MODELS = ['UNet_e2eQPAT']

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

# --- Multi-job (checkpoint) settings ---
# For the FIRST job of a multi-job run: set skip_test=True.
# For the LAST job: set resume_from to the previous job's save_dir (the full path
# including the .N<node>.j<jobid> suffix), set skip_test=False.
# Leave both at their defaults for a normal single-job run.
resume_from = None   # e.g. "/mnt/.../20260531_UNet_e2eQPAT.Naisurrey25.j1234567"
skip_test = False    # set True on all but the final job

save_dirs = {
    'EDM2': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_EDM2",
    'UNet_e2eQPAT': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_UNet_e2eQPAT",
    'UNet_wl_pos_emb': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_UNet_wl_pos_emb",
    'UNet_diffusion_ablation': f"/mnt/fast/nobackup/users/wv00017/MSOT_diffusion/MSOT_diffusion2/{datetime.now().strftime('%Y%m%d')}_UNet_diffusion_ablation"
}

for model, fold in itertools.product(MODELS, FOLDS):
    memory = '32G' if model == 'UNet_e2eQPAT' else '64G'
    partition = '3090' if model == 'UNet_e2eQPAT' else 'a100'
    time_limit = '00-06:00:00' if model == 'UNet_e2eQPAT' else '00-72:00:00'
    synthetic_or_experimental = 'experimental' if experiment in ['experimental_from_scratch', 'experimental_fine_tune'] else 'synthetic'
    synthetic_dataset = 'Digimouse' if experiment in ['Digimouse_test'] else ('Digimouse_extrusion' if experiment in ['Digimouse_extrusion_test'] else 'ImageNet')
    epochs = '1000' if model == 'EDM2' else '200'
    wandb_notes = f"{experiment}_{model}_fold{fold}"
    if experiment in ['experimental_fine_tune', 'digimouse_test', 'digimouse_extrusion_test']:
        load_best_checkpoint_from = pretrained_edm2[fold] if model == 'EDM2' else pretrained_unet_e2eqpat[fold]
    else:
        load_best_checkpoint_from = None

    command_lines = [
        "apptainer exec oras://container-registry.surrey.ac.uk/shared-containers/billy-msot_diffusion-container:latest",
        "python3 clone_and_run_msot_diffusion.py",
        "--cluster_id .N$SLURM_JOB_NODELIST.j$SLURM_JOB_ID",
        f"--save_dir {save_dirs[model]}",
        f"--synthetic_or_experimental {synthetic_or_experimental}",
        f"--synthetic_root_dir {DATASETS[synthetic_dataset]}",
        f"--experimental_root_dir {DATASETS['e2eQPAT']}",
        f"--epochs {epochs}",
        f"--model {model}",
        "--data_normalisation standard",
        f"--fold {fold}",
        f"--wandb_notes {wandb_notes}",
    ]
    if load_best_checkpoint_from:
        command_lines.append(f"--load_best_checkpoint_from {load_best_checkpoint_from}")
    if resume_from:
        command_lines.append(f"--resume_training_from {resume_from}")
    if skip_test:
        command_lines.append("--skip_test")

    command_block = " \\\n".join(f"    {line}" for line in command_lines)

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
{command_block}
    EOF
    """).strip() + "\n"

    submit_script_path = 'submit_MSOT_Diffusion.sh'
    with open(submit_script_path, 'w', encoding='utf-8') as f:
        f.write(sub_file)
    os.chmod(submit_script_path, 0o755)
    #subprocess.run(['sbatch', submit_script_path], check=True)
    