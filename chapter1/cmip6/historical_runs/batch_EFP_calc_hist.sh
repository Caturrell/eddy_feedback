#!/bin/bash
#SBATCH --job-name=batch_EFP_calc_hist  # Job name
#SBATCH --output=hist_EFP_calc_processing.out  # Standard output file
#SBATCH --error=hist_EFP_calc_processing.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=22:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/EFP_calc_historical.py