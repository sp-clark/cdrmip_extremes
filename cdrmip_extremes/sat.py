import xarray as xr
import numpy as np
import cftime


def find_crossing_years(
    gsat_da,
    window,
    gwls,
    time_dim='year',
    overshoot=True,
) -> list:
    
    # define rolling timeseries
    if time_dim == 'year':
        rolling = gsat_da.rolling(year=window,center=True,min_periods=1).mean()
    else:
        rolling = (
            gsat_da
                .groupby('time.year').mean(dim='time')
                .rolling(year=window,center=True,min_periods=1).mean()
        )
    crossing_years = {}
    if overshoot:
        # separate either side of peak warming
        peak_year = int(rolling.idxmax(dim='year'))
        pre_peak = rolling.sel(year=slice(None,peak_year))
        post_peak = rolling.sel(year=slice(peak_year+1,None))

        for gwl in gwls:
            # make mask around specified gwl for each branch and use it to identify crossing years
            pre_peak_cross = int(pre_peak.where(pre_peak>=gwl).dropna(dim='year').year[0])
            post_peak_cross = int(post_peak.where(post_peak<=gwl).dropna(dim='year').year[0])
            
            crossing_years[gwl] = {'ramp_up':pre_peak_cross,'ramp_down':post_peak_cross}
    else:
        for gwl in gwls:
            # assume we have a warming timeseries
            cross = int(rolling.where(rolling>=gwl).drop_na(dim='year').year[0])
            crossing_years[gwl] = [cross]

    da = xr.DataArray(
        [[year for year in crossing_years[gwl].values()] for gwl in crossing_years.keys()],
        coords={
            'gwl':list(crossing_years.keys()),
            'branch':['ramp_up','ramp_down']
        },
        dims=['gwl','branch']
    )

    return da

def find_matching_gwls(
    gsat_da,
    window,
    match_slice,
    time_dim='year'
):
    # define rolling timeseries
    if time_dim == 'year':
        rolling = gsat_da.rolling(year=window,center=True,min_periods=1).mean()
    else:
        rolling = (
            gsat_da
                .groupby('time.year').mean(dim='time')
                .rolling(year=window,center=True,min_periods=1).mean()
        )
    # identify end gwl
    end_gwl = np.round(
        float(rolling.sel(year=match_slice).mean(dim='year').values),
        2
    )

    # isolate ramp_up
    ramp_up = rolling.isel(year=slice(0,140))

    # identify crossing year
    exceed_year = int(ramp_up.where(ramp_up>=end_gwl).dropna(dim='year')['year'][0])

    return [end_gwl, exceed_year]

def extract_gwl_period(ds, gwl_years, window, time_dim='year'):
    """
    Extract multi-year windows of a timeseries centred around
    global warming level (GWL) crossing years.

    Parameters
    ----------
    ds : xarray.DataArray
        Yearly time series (e.g., temperature, precipitation, etc.)
        with a 'year' dimension.
    gwl_years : xarray.DataArray
        DataArray specifying the years corresponding to the crossing 
        of GWLs, with coordinates 'branch' (e.g., 'ramp_up', 'ramp_down') 
        and 'gwl' (e.g., 1.5, 2.0, 3.0).
    window : int
        Length of the window to extract, typically 21 years. 
        The window is centred on the GWL crossing year.

    Returns
    -------
    xarray.DataArray
        Concatenated DataArray containing the values of `ds` for each 
        GWL and branch within the specified time window. 
        Dimensions: ['branch', 'gwl', 'year', ...].
    """
    branches = gwl_years.branch.values
    gwls = gwl_years.gwl.values

    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')
    
    def extract_windows(years):
        """Extract windows of `ds` centered on each GWL year."""
        periods = []
        for year in years:
            start = year - (window - 1) // 2
            end = year + (window + 1) // 2
            ds_window = ds.isel(year=slice(start, end)).dropna(dim='year')
            periods.append(ds_window)
        return xr.concat(periods, dim='gwl').assign_coords(gwl=gwls)

    # Loop over branches (e.g., ramp-up, ramp-down) and extract their GWL periods
    results = [
        extract_windows(gwl_years.sel(branch=branch).values)
        for branch in branches
    ]
    
    # Combine results for both branches
    return xr.concat(results, dim='branch').assign_coords(branch=branches)

def extract_equiv_gwl_period(ds,final_gwl,exceed_year,window,time_dim='year'):

    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')
        
    # extract final 21-year period and equivalent ramp-up gwl period from ds
    ramp_up = ds.sel(year=slice(exceed_year-10,exceed_year+11)).dropna(dim='year')
    ramp_down = ds.sel(year=slice(-window,None)).dropna(dim='year')

    branches = ['ramp_up','ramp_down']

    periods = xr.concat(
        [ramp_up,ramp_down],
        dim='branch',
    ).assign_coords({'branch':branches})

    return periods

def compare_gwl_means(ds):
    """
    Function to compare the mean values across specific GWL periods
    """
    ramp_up = ds.sel(branch='ramp_up').mean(dim='year')
    ramp_down = ds.sel(branch='ramp_down').mean(dim='year')
    difference = ramp_down - ramp_up
    return {'ramp_up':ramp_up, 'ramp_down':ramp_down, 'difference':difference}