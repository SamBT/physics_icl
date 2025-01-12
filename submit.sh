#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00
#SBATCH -o slurm_logs/job_%j.out
#SBATCH -e slurm_logs/job_%j.err

# load modules
source ~/.bash_profile
mamba activate torch_gpu

cd /n/home11/sambt/hidenori/physics_icl
python train_xv_damped_noContext_v2.py $@