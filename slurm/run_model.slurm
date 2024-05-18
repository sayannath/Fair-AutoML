#!/bin/bash
####### ARC #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu2023
#SBATCH --output=fair-automl-30min.log   # Standard output and error log

####### Load necessary modules #############
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fairness-autosklearn

####### Run the script #############
sh run_evaluation.sh