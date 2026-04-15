#!/bin/bash
#SBATCH --job-name=NaN_issues  # Job name
#SBATCH --output=NaN_issues.out  # Standard output file
#SBATCH --error=NaN_issues.err  # Standard error file
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --mem=100G  # Memory (adjust based on your needs)
#SBATCH --time=72:00:00  # Max wall time (adjust based on your needs)
#SBATCH --cpus-per-task=1  # Number of cores

# Activate your environment
mamba activate eddy  # Activate the 'meddy' virtual environment

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/b-parameter/cmip6_b/hist_NaN_issues/plot_data_for_failed_models.py