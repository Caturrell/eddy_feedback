#!/bin/bash
#SBATCH --job-name=batch_IPSL  # Job name
#SBATCH --output=IPSL_hist_processing.out  # Standard output file
#SBATCH --error=IPSL_hist_processing.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=04:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/IPSL_wrangling/yearly_chunks_IPSL.py