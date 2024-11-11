"""

python /home/links/ct715/eddy_feedback/isca/scripts

"""

import numpy as np
import os

from isca import DryCodeBase, DiagTable, Experiment, Namelist, GFDL_BASE

NCORES = 16
RESOLUTION = 'T42', 40  # (horizontal resolution, levels in pressure)

# select timestep based on chosen resolution
timestep = {
    'T42': 600,
    'T85': 300,
    'T170': 150
}
dt = timestep[RESOLUTION[0]]

# a CodeBase can be a directory on the computer,
# useful for iterative development
cb = DryCodeBase.from_directory(GFDL_BASE)

# or it can point to a specific git repo and commit id.
# This method should ensure future, independent, reproducibility of results.
# cb = DryCodeBase.from_repo(repo='https://github.com/isca/isca', commit='isca1.1')

# compilation depends on computer specific settings.  The $GFDL_ENV
# environment variable is used to determine which `$GFDL_BASE/src/extra/env` file
# is used to load the correct compilers.  The env file is always loaded from
# $GFDL_BASE and not the checked out git repo.

cb.compile()  # compile the source code to working directory $GFDL_WORK/codebase

# create an Experiment object to handle the configuration of model parameters
# and output diagnostics

#--------------------------------------------------------------------------------------------------

# SET PARAMETERS

# Choose number of years
YEARS = 10
# Set equator-to-pole temperature gradient
DELH = 60.                      # default: 60K

exp_name = f'HS_{RESOLUTION[0]}_{YEARS}y_{int(DELH)}delh_t42-land'
exp = Experiment(exp_name, codebase=cb)

exp.inputfiles = [os.path.join(GFDL_BASE,'input/land_masks/era_land_t42.nc')]

#--------------------------------------------------------------------------------------------------


#Tell model how to write diagnostics
diag = DiagTable()
# diag.add_file('atmos_monthly', 30, 'days', time_units='days')
diag.add_file('atmos_daily', 1, 'days', time_units='days')

#Tell model which diagnostics to write
diag.add_field('dynamics', 'ps', time_avg=True)
diag.add_field('dynamics', 'bk')
diag.add_field('dynamics', 'pk')
diag.add_field('dynamics', 'zsurf')
diag.add_field('dynamics', 'temp', time_avg=True)

diag.add_field('dynamics', 'ucomp', time_avg=True)  
diag.add_field('dynamics', 'vcomp', time_avg=True)
diag.add_field('dynamics', 'omega', time_avg=True)  

exp.diag_table = diag

# define namelist values as python dictionary
# wrapped as a namelist object.
namelist = Namelist({
    'main_nml': {
        'dt_atmos': dt,
        'days': 30,
        'calendar': 'thirty_day',
        'current_date': [2000,1,1,0,0,0]
    },

    'atmosphere_nml': {
        'idealized_moist_model': False  # False for Newtonian Cooling.  True for Isca/Frierson
    },

    'spectral_dynamics_nml': {
        'damping_order'           : 2,                      # default: 2
        'water_correction_limit'  : 0,                      # default: 0
        'reference_sea_level_press': 1.0e5,                 # default: 101325
        'valid_range_t'           : [100., 800.],           # default: (100, 500)
        'initial_sphum'           : 0.0,                    # default: 0
        'vert_coord_option'       : 'even_sigma',           # default: 'even_sigma'
        'scale_heights': 6.0,
        'exponent': 7.5,
        'surf_res': 0.5
    },

    # configure the relaxation profile
    'hs_forcing_nml': {
        't_zero': 315.,    # temperature at reference pressure at equator (default 315K)
        't_strat': 200.,   # stratosphere temperature (default 200K)
        'delh': DELH,       # equator-pole temp gradient (default 60K)
        'delv': 10.,       # lapse rate (default 10K)
        'eps': 0.,         # stratospheric latitudinal variation (default 0K)
        'sigma_b': 0.7,    # boundary layer friction height (default p/ps = sigma = 0.7)

        # negative sign is a flag indicating that the units are days
        'ka':   -40.,      # Constant Newtonian cooling timescale (default 40 days)
        'ks':    -4.,      # Boundary layer dependent cooling timescale (default 4 days)
        'kf':   -1.,       # BL momentum frictional timescale (default 1 days)

        'do_conserve_energy':   True,  # convert dissipated momentum into heat (default True)
    },
    
    'spectral_init_cond_nml':{
         'topog_file_name': 'era_land_t42.nc', #Name of land input file, which will also contain topography if generated using Isca's `land_file_generator_fn.py' routine.
         'topography_option': 'input' #!Tell model to get topography from input file
    },

    'diag_manager_nml': {
        'mix_snapshot_average_fields': False
    },

    'fms_nml': {
        'domains_stack_size': 600000                        # default: 0
    },

    'fms_io_nml': {
        'threading_write': 'single',                         # default: multi
        'fileset_write': 'single',                           # default: multi
    }
})

exp.namelist = namelist
exp.set_resolution(*RESOLUTION)

#--------------------------------------------------------------------------------------------------


# Calculate number of months
num_months = 1 + (12 * YEARS)

#Lets do a run!
if __name__ == '__main__':
    
    print(f'\nThe experiment name is {exp_name}. Do you wish to continue?')
    
    # Pause the script and wait for user confirmation to continue
    user_input = input("Press Enter to continue, or type 'exit' to stop: ")
    if user_input.lower() == 'exit':
        print("Exiting script.")
        exit()  # Exits the script
    else:
        print("Continuing script...")
    
    exp.run(1, num_cores=NCORES, use_restart=False)
    for i in range(2, num_months):
        exp.run(i, num_cores=NCORES)  # use the restart i-1 by default
