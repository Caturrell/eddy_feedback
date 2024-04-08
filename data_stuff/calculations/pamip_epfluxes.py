import xarray as xr 
import glob 
import os 

import sys 
sys.path.append('/home/users/cturrell/documents/eddy_feedback')
import functions.eddy_feedback as ef 


def calculate_pamip_epfluxes(ta, ua, va):
    
    # Combine datasets into one
    ds = xr.Dataset(data_vars={'ta': ta.ta, 'ua': ua.ua, 'va': va.va})
    
    # Calculating EP fluxes
    print('Calculating EP fluxes...')
    ds = ef.calculate_epfluxes_ubar(ds)
    
    print('Done.')
    
    return ds 



if __name__ == '__main__':
    
    print('Importing datasets...')
    ta_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/AWI-CM-1-1-MR_3x3/ta/*')
    ua_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/AWI-CM-1-1-MR_3x3/ua/*')
    va_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/AWI-CM-1-1-MR_3x3/va/*')
    
    print(f'ta:{len(ta_files)}, ua:{len(ua_files)}, va:{len(va_files)}')
    
    
    for i in range(len(ta_files)):
        
        # import each var as dataset
        ta = xr.open_mfdataset(ta_files[i]) 
        ua = xr.open_mfdataset(ua_files[i]) 
        va = xr.open_mfdataset(va_files[i]) 
        
        
        # Start function
        print('Datasets loaded. Calculations starting...')
        ds = calculate_pamip_epfluxes(ta, ua, va) 
        
        print('Calculations complete. Now saving dataset...')
        ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/AWI-CM-1-1-MR_3x3/AWI_uvt_ep_3x3_ens{i}.nc')
         
        #  # extract each file name
        #  ta = os.path.basename(ta_files[i])
        #  ua = os.path.basename(ua_files[i])
        #  va = os.path.basename(va_files[i])
         