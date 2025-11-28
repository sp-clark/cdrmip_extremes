import xarray as xr
import numpy as np
import cftime



def global_mean(x, var="tas"):
    """
    Function to calculate cosine weighted global mean for a given variable
    """
    # Check whether input is a dataset or dataarray
    if isinstance(x, xr.Dataset):
        da = x[var]
    elif isinstance(x, xr.DataArray):
        da = x
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")

    # Weighing Data
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"

    return da.weighted(weights).mean(("lon", "lat"))

def global_surface_mean(x, var, surface):
    # Check whether input is a dataset or dataarray
    if isinstance(x, xr.Dataset):
        da = x[var]
    elif isinstance(x, xr.DataArray):
        da = x
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")

    if surface == 'land':
        weights=areacella
    elif surface == 'ocean':
        weights = areacello
    else:
        raise ValueError("surface must be 'land' or 'ocean'")

    return da.weighted(weights=weights).mean(dim=weights.dims)

def concat_branches(ds_up, ds_down):
    time_up = xr.cftime_range("0000-01-16",freq="1M",periods=12*140,calendar='noleap')
    len_down = len(ds_down.time)
    time_down = xr.cftime_range("0140-01-16",freq="1M",periods = len_down,calendar='noleap')

    ds_up = ds_up.isel(time=slice(None,12*140)).assign_coords({'time':time_up})
    ds_down = ds_down.assign_coords({'time':time_down})

    return xr.concat([ds_up,ds_down],dim='time')

def calc_anomaly(
    ds: xr.Dataset,
    ds_ref: xr.Dataset,
    reference_period: slice,
    *,
    time_dim="time",
) -> xr.Dataset:
    """
    Function to calculate anomaly with respect to 
    reference period. Assumes child dataset is already
    concatenated to parent.
    """
    # select reference slice and mean
    if time_dim not in ds.dims:
        raise ValueError(f"{time_dim} is not a dimension of this dataset. "
                         f"Available dims: {list(ds.dims)}")
    # calculate reference mean
    ref = ds_ref.sel({time_dim:reference_period})
    ref = ref.mean(dim=time_dim)

    # subtract reference from ds
    anom = ds - ref
    return anom

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


