#!/bin/bash

#SBATCH --partition=test
#SBATCH --job-name=pamip_regrid
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=04:00:00

module load jaspy
echo 'Python loaded.'

# echo 'Regridding script running...'
# python /home/users/cturrell/documents/eddy_feedback/data_stuff/regridding/regridding.py

echo 'EP Flux calculations script running...'
python /home/users/cturrell/documents/eddy_feedback/data_stuff/calculations/calc_pamip_epfluxes.py