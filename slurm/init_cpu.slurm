#!/bin/bash
####### ARC #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=48G
#SBATCH --partition=cpu2023
#SBATCH --output=adult.log   # Standard output and error log

####### Load necessary modules #############
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fairness-autosklearn

####### Run the script #############
cd ..
cd evaluation/adult
./run_adult.sh