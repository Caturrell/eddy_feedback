import xarray as xar
# import aostools.climate as aoscli
import numpy as np
import os
import pdb
# import xcdat
import logging
# import eddy_feedback_functions as eff
# import eddy_plotting_functions as epf
from SIT_search_for_hist import find_ensemble_list, find_ensemble_list_multi_var
from tqdm import tqdm
import xesmf as xe

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all existing handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

exp_type='cmip6'

if exp_type=='cmip6':
    
    
    ## CanESM5 attempt?
    # base_dir = '/badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/piControl/r1i1p1f1/day/'
    # grid_part = '/gn/latest/'
    # exp_name='CanESM5_piControl'
    subtract_annual_cycle=True
    # start_month = '5201'
    # end_month   = '5210'
    level_type='6hrPlevPt'
    # files = [f'{base_dir}{var_name}{grid_part}/{var_name}_day_CanESM5_piControl_r1i1p1f1_gn_{start_month}0101-{end_month}1231.nc' for var_name in ['ua', 'va', 'ta']]

    mip_id = 'CMIP'
    base_dir_badc = f'/badc/cmip6/data/CMIP6/{mip_id}/'
    base_dir_output = '/gws/nopw/j04/arctic_connect/sthomson/efp_6hourly_processed_data/1000hPa_100hPa_slice_inner2/'

    version_name='latest'
    experiment = 'historical'

    total_time_span_required = 10.0 #specify number of years we want to analyse

    force_recalculate=False

    var_name_dict = {
        'ua':{'data_type':'6hrPlevPt'}, #Pt here means snapshots, whereas 6hrPlev would be 6 hourly averages (see cell_methods here https://github.com/PCMDI/cmip6-cmor-tables/blob/main/Tables/CMIP6_6hrPlevPt.json vs here https://github.com/PCMDI/cmip6-cmor-tables/blob/main/Tables/CMIP6_6hrPlev.json)
        'va':{'data_type':'6hrPlevPt'},
        # 'ta':{'data_type':'6hrPlevPt'},
    }

    logging.info('finding all available data')
    models_by_var = find_ensemble_list_multi_var(base_dir_badc, var_name_dict, experiment)
    available_models_dict_to_use=models_by_var['ua']
    logging.info('done finding all available data')
    #loop over each model 

    one_member_files_dict = {}    
    start_month_dict = {}
    end_month_dict = {}
    dodgy_model_list = []
    weird_model_list = []

    for model_name in available_models_dict_to_use.keys():

        files_for_model = []
        n_files_required_list = []
        end_year_list = []
        n_years_available_list = []
        files_for_model_dict = {}
        start_date_for_model_dict = {}
        end_date_for_model_dict = {}

        for var_name_to_check in var_name_dict.keys():
            ens_list = models_by_var[var_name_to_check][model_name]['ens_ids']
            grid_list = models_by_var[var_name_to_check][model_name]['grid_list']
            ens_list = models_by_var[var_name_to_check][model_name]['ens_ids']

            if len(ens_list)==1:
                member_choice = ens_list[0]
            elif 'r1i1p1f1' in ens_list:
                member_choice = 'r1i1p1f1'
            else:
                raise NotImplementedError(f'Not sure which ens_id to choose for {model_name}')

            if len(grid_list)==1:
                grid_choice = grid_list[0]
            elif 'gn' in grid_list:
                grid_choice='gn'
            elif 'gr' in grid_list:
                grid_choice='gr'                
            else:
                raise NotImplementedError(f'Not sure what type of grid to choose for {model_name}')

            logging.info(f'Using {grid_choice} and ens_id={member_choice} for {model_name} and var name {var_name_to_check}')


            if member_choice in ens_list and grid_choice in grid_list:
                files_for_model_var = [file for file in models_by_var[var_name_to_check][model_name]['files'] if member_choice in file and f'/{grid_choice}/' in file]

                start_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:4] for file_name in files_for_model_var]
                end_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:4] for file_name in files_for_model_var]                

                start_date = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:8] for file_name in files_for_model_var]
                end_date = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:8] for file_name in files_for_model_var]                

                len_each_file = np.asarray(np.int32(end_year)) - np.asarray(np.int32(start_year))
                
                any_dt = np.int32(start_year[1:])- np.int32(end_year[0:-1])

                if end_date[0][4:8]=='1231':
                    len_each_file = len_each_file + 1

                n_files_required = np.int32(np.ceil(total_time_span_required/len_each_file[0]))
                n_files_required_float = total_time_span_required/len_each_file[0]

                continuous_timeseries = True
                if np.any(any_dt>1.):
                    logging.warning(f'there appears to be a gap in the timeseries for model {model_name}')
                    if np.any(any_dt[0:n_files_required]>1):
                        logging.warning(f'This is going to affect your calculation')
                        continuous_timeseries=False

                if n_files_required>len(files_for_model_var):
                    logging.info(f'Insufficient files for {model_name}')
                    dodgy_model_list.append(model_name)
                    n_files_required = len(files_for_model_var)

                n_files_required_list.append(n_files_required)
                end_year_list.append(end_year[n_files_required-1])

                n_years_available_list.append(np.sum(len_each_file))

                files_for_model_dict[var_name_to_check] = files_for_model_var[0:n_files_required]
                start_date_for_model_dict[var_name_to_check] = start_date[0:n_files_required]
                end_date_for_model_dict[var_name_to_check] = end_date[0:n_files_required]

        # if not np.all(np.asarray(n_files_required_list) == n_files_required_list[0]):
            # pdb.set_trace()
        if not np.all(np.asarray(end_year_list) == end_year_list[0]):
            end_year_values = [float(end_year_val) for end_year_val in end_year_list]
            end_month_dict[model_name]   = str(np.int64(np.min(np.asarray(end_year_values))))
        else:
            end_month_dict[model_name]   = end_year[n_files_required-1]

        start_month_dict[model_name] = start_year[0]

        n_files_per_var = []
        for var_name_to_check in var_name_dict.keys():
            n_files_per_var.append(len(files_for_model_dict[var_name_to_check]))
        if not np.all(np.asarray(n_files_per_var)==n_files_per_var[0]) or not continuous_timeseries:
            weird_model_list.append(model_name)

        one_member_files_dict[model_name] = {'files':files_for_model_dict, 'start_date':start_date_for_model_dict, 'end_date':end_date_for_model_dict}

        logging.info(n_years_available_list)
        
else:
    raise NotImplementedError(f'no valid exp type configured for {exp_type}')

omega = 2.*np.pi/86400.
a0 = 6371000.
do_individual_plots = True
do_individual_corr_plots = True
do_big_TEM_plot = True
do_heatmap_correlations_plot = True
do_eof_plots = True

force_ep_flux_recalculate = False
monthly_too=True

if exp_type=='cmip6':
    model_list = [key for key in one_member_files_dict.keys()]
else:
    model_list = [exp_type]

logging.info(model_list)
logging.info('dodgy models')
logging.info(np.unique(dodgy_model_list))
logging.info('weird models')
weird_model_list = [model for model in weird_model_list if model not in dodgy_model_list]
logging.info(np.unique(weird_model_list))
logging.info('good models')
good_model_list = [model for model in model_list if model not in weird_model_list and model not in dodgy_model_list]
logging.info(good_model_list)

# good_model_list = good_model_list[2:] #first 2 have completed

# pdb.set_trace()

# good_model_list = ['CMCC-ESM2', 'EC-Earth3-CC', 'MPI-ESM-1-2-HAM', 'IPSL-CM5A2-INCA', 'IPSL-CM6A-LR', 'KIOST-ESM', 'NorESM2-LM', 'NorESM2-MM',]
# good_model_list = ['IPSL-CM5A2-INCA', 'NorESM2-LM', 'IPSL-CM6A-LR',  ]#'NorESM2-MM', ]
# pdb.set_trace()
# good_model_list = ['NorESM2-MM', ]


# model_list = ['CanESM5', 'IPSL-CM6A-LR', 'CNRM-CM6-1']
# model_list = ['MIROC6']

# good_model_list = good_model_list
# good_model_list = ['EC-Earth3', 'GFDL-CM4', 'MRI-ESM2-0']
pdb.set_trace()
for model_name in good_model_list:

    logging.info(f'Now looking at {model_name}')
    try:
    # if True:
        if exp_type=='cmip6':
            start_month = start_month_dict[model_name]
            end_month   = end_month_dict[model_name]
            files = one_member_files_dict[model_name]['files']
            start_month_list_by_files = one_member_files_dict[model_name]['start_date']['ua']
            end_month_list_by_files = one_member_files_dict[model_name]['end_date']['ua']   

            logging.info(f'Reading {len(files['ua'])} files for {model_name} from {start_month} to {end_month}')

            start_month_val = int(start_month)
            end_month_val = int(end_month)        

            if end_month_val - start_month_val>total_time_span_required:
                end_month=str(int(start_month_val+total_time_span_required-1))
                time_slice = slice(f'{start_month}-01-01', f'{end_month}-12-31')
                slice_time=True
            else:
                slice_time=False

        plot_dir = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/'

        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        yearly_data_dir = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/yearly_data/'

        if not os.path.isdir(yearly_data_dir):
            os.makedirs(yearly_data_dir)

        for year_idx, start_date_val in tqdm(enumerate(start_month_list_by_files)):    

            end_date_val = end_month_list_by_files[year_idx]

            output_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_epflux.nc'
            output_day_av_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_daily_averages.nc'
            output_month_av_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_monthly_averages.nc'

            files_for_year = []
            for var_name_to_check in var_name_dict.keys():
                files_for_year = files_for_year + [files[var_name_to_check][year_idx]]

            # def preprocess(ds):
            #     """Subset dataset to years 2000-2010."""
            #     return ds.sel(time=slice("1870-01-01", "1880-12-31"))

            if not os.path.isfile(output_file) or force_ep_flux_recalculate or not os.path.isfile(output_day_av_file):
                logging.info('opening model data files')
                time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
                dataset = xar.open_mfdataset(files_for_year, decode_times=time_coder,
                                        parallel=False, join='inner' )#chunks={'time': 10, 'pfull':28,})

                if np.all(dataset.plev.diff('plev')>0.):
                    pfull_slice = slice(100., 1000.)
                else:
                    pfull_slice = slice(1000., 100.)

                if 'units' in dataset['plev'].attrs.keys():
                    if dataset['plev'].attrs['units']=='Pa':
                        dataset['plev'] = dataset['plev']/100.
                elif dataset['plev'].max().values>1000.:
                        dataset['plev'] = dataset['plev']/100.                    

                do_pfull_slice = True

                if dataset.plev.sel(plev=pfull_slice).shape[0]<3:
                    dataset.close()
                    logging.info('re-opening model data files due to too few pfull levels')
                    dataset = xar.open_mfdataset(files_for_year, decode_times=time_coder,
                                            parallel=False )#chunks={'time': 10, 'pfull':28,}) 
                    do_pfull_slice = False
                    if dataset.plev.shape[0]<3:
                        # pdb.set_trace()
                        orig_plev = dataset.dims['plev'].values
                        new_plev  = np.zeros(3)
                        new_plev[:orig_plev.shape[0]] = orig_plev
                        new_plev[orig_plev.shape[0]:] = np.zeros(3-orig_plev.shape[0]) + np.nan
                        # pdb.set_trace()

                inconsistent_grid = False
                if dataset.lat.shape[0]==0:
                    logging.info('inconsistent lat values across files')
                    inconsistent_grid = True
                if dataset.lon.shape[0]==0:
                    logging.info('inconsistent lon values across files')
                    inconsistent_grid = True

                if inconsistent_grid:
                    logging.info('need to identify which file has an inconsistent grid')
                    ds_list = []
                    lat_list = []
                    lon_list = []
                    for file_in_year in files_for_year:
                        ds_temp = xar.open_mfdataset(file_in_year, decode_times=time_coder, parallel=False)
                        ds_list.append(ds_temp)
                        lat_list.append(ds_temp.lat.values)
                        lon_list.append(ds_temp.lon.values)

                    # does the first one match any of the others?
                    for file_idx in range(len(files_for_year)):
                        arr_lat_match_len = [lat_list_val.shape[0] == lat_list[file_idx].shape[0] for lat_list_val in lat_list]
                        arr_lon_match_len = [lon_list_val.shape[0] == lon_list[file_idx].shape[0] for lon_list_val in lon_list]        
                        if np.any(np.logical_not(arr_lat_match_len)):              
                            odd_one_out_lat_len = np.where(np.logical_not(arr_lat_match_len))[0]
                            if odd_one_out_lat_len.shape[0]==1:
                                logging.info(f'uniquely identified that it is the {odd_one_out_lat_len[0]} index file that is odd')
                                same_one_idx = np.where(arr_lat_match_len)[0][0]
                                odd_one_idx = odd_one_out_lat_len[0]

                                regridded_file_name = files_for_year[odd_one_idx].split('/')[-1]
                                final_regrid_file_name = f'{yearly_data_dir}/{regridded_file_name}'
                                if not os.path.isfile(final_regrid_file_name):
                                    logging.info('regridded file does not exist - calculating')
                                    ds_out = xar.Dataset({'lat': (['lat'], ds_list[same_one_idx].lat.values),
                                                        'lon': (['lon'], ds_list[same_one_idx].lon.values),
                                                        }
                                                    )

                                    #       ds_out.attrs = ens_mean_dataset.attrs
                                    logging.info('setting up regrid')
                                    regridder = xe.Regridder(ds_list[odd_one_idx], ds_out, 'bilinear', ignore_degenerate=True)
                                    # regridder.clean_weight_file()

                                    ds_out = regridder(ds_list[odd_one_idx])
                                    logging.info('writing regrid to file')

                                    ds_out.to_netcdf(final_regrid_file_name)
                                    ds_out.close()
                                else:
                                    logging.info('regridded file does exist - opening')

                                ok_files = [file_name for file_name in files_for_year if file_name!=files_for_year[odd_one_idx]]
                                ok_files.append(final_regrid_file_name)
                                logging.info('reopening file')
                                dataset = xar.open_mfdataset(ok_files, decode_times=time_coder,
                                            parallel=False)
                                logging.info('should now have a proper size array')
                                pass

                        else:
                            arr_lat_match = [lat_list_val == lat_list[file_idx] for lat_list_val in lat_list]
                            arr_lon_match = [lon_list_val == lon_list[file_idx] for lon_list_val in lon_list]
                            if np.where(np.asarray(arr_lat_match))[0].shape[0]>1:
                                logging.info('the first file matches at least one of the others')
                                odd_lat_idx_out = np.where(not arr_lat_match)[0]
                                odd_lon_idx_out = np.where(not arr_lon_match)[0]                            
                            raise NotImplementedError('Help')                    
                    
                logging.info('COMPLETE')
                
                if exp_type=='jra55':

                    logging.info('opening OLD JRA-55 dataset to grab time bounds')
                    old_dataset = xar.open_mfdataset(['/disca/share/sit204/jra_55/1958_2016/atmos_daily_uvtw.nc'], decode_times=time_coder,
                                            parallel=True, chunks={'time': 50})    
                    
                    dataset['time_bnds'] = old_dataset['time_bnds']    
                    logging.info('finished adding OLD JRA-55 dataset time bounds')    

                    dataset = dataset.rename({
                        'u':'ucomp',
                        'v':'vcomp',
                        't':'temp',
                        'level':'pfull',
                    })

                elif exp_type=='cmip6':
                    dataset = dataset.rename({
                        'ua':'ucomp',
                        'va':'vcomp',
                        'plev':'pfull',
                        })
                    
                    if 'ta' in dataset.keys():
                        dataset = dataset.rename({
                            'ta':'temp',
                            })                                                

                    if 'units' in dataset['pfull'].attrs.keys():
                        if dataset['pfull'].attrs['units']=='Pa':
                            dataset['pfull'] = dataset['pfull']/100.
                    elif dataset['pfull'].max().values>1000.:
                            dataset['pfull'] = dataset['pfull']/100.
                    
                    chunk_size_var_dict = {}
                    ds_coords = [val for val in dataset.coords.keys()]
                    ds_vars = [val for val in dataset.variables.keys() if val not in ds_coords and 'bnds' not in val and 'bounds' not in val]
                    for var_name in ds_vars:
                        chunk_size_var_dict[var_name] = dataset[var_name].chunksizes
                    all_chunks_equal = [chunk_size_var_dict[var_name] == chunk_size_var_dict[ds_vars[0]] for var_name in ds_vars]
                    if not np.all(all_chunks_equal):
                        dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0], 'lat':dataset.lat.shape[0], 'lon':dataset.lon.shape[0]})

                if 'udt_rdamp' in dataset.data_vars.keys():
                    include_udt_rdamp=True
                else:
                    include_udt_rdamp=False    

                if np.all(dataset.pfull.diff('pfull')>0.):
                    pfull_slice = slice(100., 1000.)
                else:
                    pfull_slice = slice(1000., 100.)

                if do_pfull_slice:
                    dataset = dataset.sel(pfull=pfull_slice)

                chunk_size_var_dict = {}
                ds_coords = [val for val in dataset.coords.keys()]
                ds_vars = [val for val in dataset.variables.keys() if val not in ds_coords and 'bnds' not in val and 'bounds' not in val]
                for var_name in ds_vars:
                    chunk_size_var_dict[var_name] = dataset[var_name].chunksizes
                all_chunks_equal = [chunk_size_var_dict[var_name] == chunk_size_var_dict[ds_vars[0]] for var_name in ds_vars]
                if not np.all(all_chunks_equal):
                    dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0], 'lat':dataset.lat.shape[0], 'lon':dataset.lon.shape[0]})

                if min(chunk_size_var_dict[ds_vars[0]]['pfull'])<3:
                    logging.warning('rechunking pfull to allow edge-order 2 vert deriv')
                    dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0]})

                if slice_time:
                    dataset = dataset.sel(time=time_slice)

                logging.info(f'This dataset has {dataset.pfull.shape[0]} plevels, {dataset.lat.shape[0]} lat points , {dataset.time.shape[0]} time points and {dataset.lon.shape[0]} longitude points')

                if 'temp' not in dataset.keys():
                    dataset['temp'] = xar.zeros_like(dataset['ucomp'])+np.nan

                # try:
                epflux_ds = eff.ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega, a0)
                # except Exception as e:
                    # logging.info(f"An error occurred: {e}")
                    # pdb.set_trace()

                logging.info('merging dataset')
                dataset = xar.merge([dataset, epflux_ds])
                logging.info('FINISHED merging dataset')

                dataset, duplicates_found = eff.check_for_duplicate_times(dataset)

                if duplicates_found:
                    dataset, duplicates_found_2 = eff.check_for_duplicate_times(dataset) #run a second time to check if successful
                    assert(not duplicates_found_2) #make sure that duplicates were not found a second time, if not throw error
            else:
                dataset = None
                epflux_ds = None

            dataset_daily = eff.daily_average(dataset, output_day_av_file, force_ep_flux_recalculate, monthly_too=monthly_too, monthly_output_file=output_month_av_file)

            dataset_daily.close()

            if epflux_ds is not None:
                epflux_ds.close()

            if dataset is not None:
                dataset.close()


    except Exception as e:
        logging.info(f'failed for {model_name} with reason:')
        logging.info(e)
        logging.info('continuing')