import xarray as xr

def calculate_ef_parameter(ds, reanalysis=True):

    """ 
    Input: Xarray dataset containing u and div1
            - div1 calculated using aostools (Martin Jucker)

    Output: Xarray dataArray of Eddy Feedback Parameter (EFP)
    """

    ## CONDITIONS

    # Reduce dataset to 200-600hPa
    ds = ds.where( ds.level >= 200., drop=True ) 
    ds = ds.where( ds.level <= 600., drop=True ) 

    # choose northern hemisphere
    ds = ds.where( ds.lat >= 0, drop=True )


    #-------------------------------------------------------------------

    ## SET UP TIME
    
    # set variables and save them
    ubar = ds.u.mean(('lon'))
    div1 = ds.div1
    
    if reanalysis: 
        # separate time into annual means
        # and use .load() to force the calculation now
        ubar = ubar.groupby('time.year').mean('time').load()
        div1 = div1.groupby('time.year').mean('time').load()
    else:
        # separate time into annual means
        ubar = ubar.load()
        div1 = div1.load()

    # calculate Pearson's correlation
    R = xr.corr(ubar, div1, dim='year')

    # calculate correlation squared (R^2)
    R = R**2

    # calculate average of levels
    R = R.mean(('level'))

    return R