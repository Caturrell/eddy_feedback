#!/bin/bash
#SBATCH --job-name=batch_hist_era5  # Job name
#SBATCH --output=era5_hist_processing2.out  # Standard output file
#SBATCH --error=era5_hist_processing2.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=96:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/era5_250-850_calc_efp_b.py