#!/bin/bash
#SBATCH --job-name=efp_calc_hist  # Job name
#SBATCH --output=efp_calc_hist.out  # Standard output file
#SBATCH --error=efp_calc_hist.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mem=20G  # Memory (adjust based on your needs)
#SBATCH --time=23:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/daily-efp_hist/calc_daily-efp_hist.py