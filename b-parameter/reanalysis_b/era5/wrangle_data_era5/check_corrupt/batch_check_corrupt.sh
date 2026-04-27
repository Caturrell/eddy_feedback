#!/bin/bash
#SBATCH --job-name=check_corrupt_era5
#SBATCH --output=check_corrupt.out
#SBATCH --error=check_corrupt.err
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=5G            # reduced: chunked single-file processing needs far less
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1

# Activate your environment
mamba activate eddy

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/wrangle_data_era5/check_corrupt/check_corrupt.py