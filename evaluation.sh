#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 2
#SBATCH -t 0-01:00
#SBATCH -o slurm_logs/job_%j.out
#SBATCH -e slurm_logs/job_%j.err

# load modules
source ~/.bash_profile
mamba activate torch_gpu

cd /n/home11/sambt/hidenori/physics_icl
python evaluate_models.py $@