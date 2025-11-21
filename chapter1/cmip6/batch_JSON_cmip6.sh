#!/bin/bash
#SBATCH --job-name=batch_cmip6  # Job name
#SBATCH --output=JSON_cmip6.out  # Standard output file
#SBATCH --error=JSON_cmip6.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=32G  # Memory (adjust based on your needs)
#SBATCH --time=48:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment


python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/save_to_JSON.py
# python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/json_to_csv.py