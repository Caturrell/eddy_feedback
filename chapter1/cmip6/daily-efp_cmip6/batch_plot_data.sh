#!/bin/bash
#SBATCH --job-name=plot_data  # Job name
#SBATCH --output=plot_data.out  # Standard output file
#SBATCH --error=plot_data.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=200G  # Memory (adjust based on your needs)
#SBATCH --time=2:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
# python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/daily-efp_cmip6/missing_time_SIT.py

python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/daily-efp_cmip6/missing_time_data.py