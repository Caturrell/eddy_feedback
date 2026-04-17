#!/bin/bash
#SBATCH --job-name=batch_hist  # Job name
#SBATCH --output=hadgem_hist_processing.out  # Standard output file
#SBATCH --error=hadgem_hist_processing.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=48:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/b-parameter/cmip6_b/hist_calc_efp_b.py