#!/bin/bash
#SBATCH --job-name=is_corrupt  # Job name
#SBATCH --output=is_corrupt.out  # Standard output file
#SBATCH --error=is_corrupt.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=32G  # Memory (adjust based on your needs)
#SBATCH --time=48:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Load the necessary modules
# module load python

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/check_corrupt/is_corrupt.py

