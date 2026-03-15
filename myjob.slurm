#!/usr/bin/env bash

# Slurm configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=test.log
#SBATCH --time=48:00:00

python -u power_samp_gpqa.py --mcmc_steps=10 --temp=0.25 --model=qwen --cot True
