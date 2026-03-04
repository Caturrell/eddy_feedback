#!/bin/bash
#SBATCH --job-name=batch_IPSL_hist  # Job name
#SBATCH --output=calc_IPSL_ep-fluxes.out  # Standard output file
#SBATCH --error=calc_IPSL_ep-fluxes.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=23:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/IPSL_wrangling/process_IPSL_hist_end2015.py