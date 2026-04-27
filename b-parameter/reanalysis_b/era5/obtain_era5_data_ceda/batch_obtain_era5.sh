#!/bin/bash
#SBATCH --job-name=obtain_era5
#SBATCH --output=obtain_era5_%a.out
#SBATCH --error=obtain_era5_%a.err
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-2%3

# VARIABLES[0]=u  VARIABLES[1]=v  VARIABLES[2]=t

mamba activate eddy

python /home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/obtain_era5_data_ceda/ceda_save_yearly_era5.py
