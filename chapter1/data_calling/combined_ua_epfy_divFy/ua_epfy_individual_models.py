import xarray as xr
# import jsmetrics as js
from pathlib import Path
import os
import numpy as np

import functions.data_wrangling as data
import functions.eddy_feedback as ef

import warnings
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

path = '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC'
models = ['CESM1-WACCM-SC', 'CNRM-CM6-1', 'EC-EARTH3', 'IPSL-CM6A-LR', 'OpenIFS-159']

# cnrm_ua = xr.open_mfdataset(
#     '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/ua/CNRM-CM6-1/*.nc',
#     chunks={'time': 31},
#     combine='nested',
#     concat_dim='ens_ax'
#     )

# cnrm_epfy = xr.open_mfdataset(
#     '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/epfy/CNRM-CM6-1/*.nc',
#     chunks={'time': 31},
#     combine='nested',
#     concat_dim='ens_ax'
#     )

# cnrm_epfy = cnrm_epfy.sel(lon=0)

# # match pressure levels to smaller dataset
# if len(cnrm_epfy.level) > len(cnrm_ua.level):
#     cnrm_epfy = cnrm_epfy.sel( level = cnrm_ua.level.values )
# else:
#     cnrm_ua = cnrm_ua.sel( level = cnrm_epfy.level.values )
    
# # Save the processed dataset
# cnrm_save_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC'
# cnrm_output_file = os.path.join(cnrm_save_dir, f'{models[1]}_1.1_u_ubar_epfy_divFy.nc')
# print(str(cnrm_output_file))
# print(models[1])
    
# # Merge datasets and calculate divFy
# cnrm_ds = xr.Dataset({'ua': cnrm_ua.ua, 'epfy': cnrm_epfy.epfy})
# cnrm_ds = ef.calculate_divFphi(cnrm_ds)

# cnrm_efp_nh = ef.calculate_efp(cnrm_ds, data_type='pamip')
# cnrm_efp_sh = ef.calculate_efp(cnrm_ds, data_type='pamip', calc_south_hemis=True)
# print(f'NH: {cnrm_efp_nh}\nSH: {cnrm_efp_sh}')

# cnrm_ds.to_netcdf(cnrm_output_file)




# ipsl_ua = xr.open_mfdataset(
#     '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/ua/IPSL-CM6A-LR/*.nc',
#     chunks={'time': 31}
#     )

# ipsl_epfy = xr.open_mfdataset(
#     '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/epfy/IPSL-CM6A-LR/*.nc',
#     chunks={'time': 31}
#     )

# ipsl_ua = ipsl_ua.rename({'record': 'ens_ax'})

# # match pressure levels to smaller dataset
# if len(ipsl_epfy.level) > len(ipsl_ua.level):
#     ipsl_epfy = ipsl_epfy.sel( level = ipsl_ua.level.values )
# else:
#     ipsl_ua = ipsl_ua.sel( level = ipsl_epfy.level.values )
    
# # Save the processed dataset
# ipsl_save_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC'
# ipsl_output_file = os.path.join(ipsl_save_dir, f'{models[3]}_1.1_u_ubar_epfy_divFy.nc')
# print(str(ipsl_output_file))
# print(models[3])
    
# # Merge datasets and calculate divFy
# ipsl_ds = xr.Dataset({'ua': ipsl_ua.ua, 'epfy': ipsl_epfy.epfy})
# ipsl_ds = ef.calculate_divFphi(ipsl_ds)

# ipsl_efp_nh = ef.calculate_efp(ipsl_ds, data_type='pamip')
# ipsl_efp_sh = ef.calculate_efp(ipsl_ds, data_type='pamip', calc_south_hemis=True)
# print(f'NH: {ipsl_efp_nh}\nSH: {ipsl_efp_sh}')

# ipsl_ds.to_netcdf(ipsl_output_file)


# ec_ua = xr.open_mfdataset(
#     '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/ua/EC-EARTH3/*.nc',
#     chunks={'time': 31},
#     combine='nested',
#     concat_dim='ens_ax'
#     )
# ec_epfy = xr.open_mfdataset(
#     '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/epfy/EC-EARTH3/*.nc',
#     chunks={'time': 31},
#     combine='nested',
#     concat_dim='ens_ax'
#     )
# ec_epfy = ec_epfy.mean('lon')

# # match pressure levels to smaller dataset
# if len(ec_epfy.level) > len(ec_ua.level):
#     ec_epfy = ec_epfy.sel( level = ec_ua.level.values )
# else:
#     ec_ua = ec_ua.sel( level = ec_epfy.level.values )
    
# # Save the processed dataset
# save_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC'
# output_file = os.path.join(save_dir, f'{models[2]}_1.1_u_ubar_epfy_divFy.nc')
# print(str(output_file))
# print(models[2])
    
# # Merge datasets and calculate divFy
# ds = xr.Dataset({'ua': ec_ua.ua, 'epfy': ec_epfy.epfy})
# ds = ef.calculate_divFphi(ds)

# efp_nh = ef.calculate_efp(ds, data_type='pamip')
# efp_sh = ef.calculate_efp(ds, data_type='pamip', calc_south_hemis=True)
# print(f'NH: {efp_nh}\nSH: {efp_sh}')

# ds.to_netcdf(output_file)

open159_tem = xr.open_mfdataset(
    '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/epfy/OpenIFS-159/*.nc',
    chunks={'time': 31},
    combine='nested',
    concat_dim='ens_ax'
    )
open159_tem = open159_tem.sel(lon=0)

open159_ua = xr.open_mfdataset(
    '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/ua/OpenIFS-159/*.nc',
    chunks={'time': 31},
    combine='nested',
    concat_dim='ens_ax'
    )

# Merge datasets and calculate divFy
open159_ds = xr.Dataset({'ubar': open159_tem.ua, 'epfy': open159_tem.epfy})
open159_ds = ef.calculate_divFphi(open159_ds)
open159_ds['u'] = open159_ua.ua

print(open159_ds)
print(open159_ds.data_vars)

# Save the processed dataset
open159_save_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC'
open159_output_file = os.path.join(open159_save_dir, f'{models[4]}_1.1_u_ubar_epfy_divFy.nc')
print(str(open159_output_file))
print(models[4])

open159_efp_nh = ef.calculate_efp(open159_ds, data_type='pamip')
open159_efp_sh = ef.calculate_efp(open159_ds, data_type='pamip', calc_south_hemis=True)
print(f'NH: {open159_efp_nh}\nSH: {open159_efp_sh}')

open159_ds.to_netcdf(open159_output_file)
