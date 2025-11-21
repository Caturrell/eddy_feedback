#!/bin/bash
#SBATCH --job-name=batch_cmip6  # Job name
#SBATCH --output=save_cmip6_efp_data.out  # Standard output file
#SBATCH --error=save_cmip6_efp_data.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=32G  # Memory (adjust based on your needs)
#SBATCH --time=48:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/save_cmip6_efp_data.py