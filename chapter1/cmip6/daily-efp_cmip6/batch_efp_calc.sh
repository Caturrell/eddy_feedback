#!/bin/bash
#SBATCH --job-name=efp_calc  # Job name
#SBATCH --output=efp_calc.out  # Standard output file
#SBATCH --error=efp_calc.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=48:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/daily-efp_cmip6/calc_daily-efp_annual_cycle.py