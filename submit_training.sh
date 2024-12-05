#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --mem=32G
#SBATCH -o slurm_logs/job_%j.out
#SBATCH -e slurm_logs/job_%j.err

# load modules
source ~/.bash_profile
mamba activate torch_gpu

cd /n/home11/sambt/hidenori/icl_physicsLaws

python train.py