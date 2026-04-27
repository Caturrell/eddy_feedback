#!/bin/bash
#SBATCH --job-name=era5_calc_epf
#SBATCH --output=era5_calc_epf_%a.out
#SBATCH --error=era5_calc_epf_%a.err
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=3-40

# task_id 3 → year 1979, task_id 4 → year 1980, …, task_id 40 → year 2016

mamba activate eddy

python /home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/obtain_era5_data_ceda/ep_fluxes_daily_averages.py
