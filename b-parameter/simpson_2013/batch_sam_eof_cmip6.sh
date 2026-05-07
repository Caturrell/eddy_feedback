#!/bin/bash
#SBATCH --job-name=batch_sam_eof  # Job name
#SBATCH --output=sam_eof_cmip6_hist.out  # Standard output file
#SBATCH --error=sam_eof_cmip6_hist.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=48:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/b-parameter/simpson_2013/u_SAM_cmip6_hist.py