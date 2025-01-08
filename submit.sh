#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 0-01:00
#SBATCH --mem=16G
#SBATCH -o slurm_logs/job_%j.out
#SBATCH -e slurm_logs/job_%j.err

# load modules
source ~/.bash_profile
mamba activate torch_gpu