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
    ds = ef.calculate_epfluxes_ubar(ds, pamip_data=True)
    
    return ds 



if __name__ == '__main__':
    
    print('Importing datasets...')
    ta_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/ta/*')
    ua_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/ua/*')
    va_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/va/*')
    
    print(f'ta:{len(ta_files)}, ua:{len(ua_files)}, va:{len(va_files)}')
    
    
    for i in range(len(ta_files)):
        
        # import each var as dataset
        ta = xr.open_mfdataset(ta_files[i]) 
        ua = xr.open_mfdataset(ua_files[i]) 
        va = xr.open_mfdataset(va_files[i]) 
        
        
        # Start function
        print(f'Datasets ({i+1}) loaded. Calculations starting...')
        ds = calculate_pamip_epfluxes(ta, ua, va) 
        
        print('Calculations complete. Now saving dataset...')
        ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/all/CanESM5_uvt_epfluxes_3x3_ens{i+1}.nc')
        
        print(f'CanESM5 dataset saved. {i+1} iteration completed.')
         
        
    print('PROGRAM COMPLETED.') 
         