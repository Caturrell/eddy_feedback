#!/bin/bash
#SBATCH --job-name=batch_daily_hist  # Job name
#SBATCH --output=calc_daily_ep-fluxes.out  # Standard output file
#SBATCH --error=calc_daily_ep-fluxes.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=96:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/daily-efp_hist/calc_daily_ep-fluxes_historical.py