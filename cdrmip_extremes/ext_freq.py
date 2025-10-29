import xarray as xr
import numpy as np
import cftime


def monthly_extrema(da):
    """
    Returns the months of maximum and minimum temperature for each grid cell
    """
    grouped = da.groupby('time.month').mean('time')
    return grouped.idxmax('month'), grouped.idxmin('month')

def monthly_extrema_dt(ds:xr.Dataset) -> xr.Dataset:
    if 'tas' in ds:
        max_months, min_months = monthly_extrema(ds)
        extrema = xr.concat([max_months,min_months],dim='extrema').assign_coords({'extrema':['max','min']}).rename({'tas':'month'})
        return extrema
    else:
        return ds

def extreme_month_means(months,ds):
    """
    Returns gridded xarray specifying mean tas values for the months specified in 
    'months' - will be either the month of maximum or minimum temperature for each grid cell
    """
    return ds.where(ds.time.dt.month==months).mean(dim='time')

def extreme_month_std_dev(months,ds):
    """
    Returns gridded xarray specifying standard deviation in tas values for the months specified in 
    'months' - will be either the month of maximum or minimum temperature for each grid cell
    """
    return ds.where(ds.time.dt.month==months).std(dim='time')

def extreme_month_stat(ds,months,stat):
    ds_month = ds.where(ds.time.dt.month==months)
    if stat == 'mean':
        return ds_month.mean(dim='time')
    elif stat == 'std':
        return ds_month.std(dim='time')
    else:
        raise ValueError("stat must be 'mean' or 'std'")
    

def heat_extreme_thresholds(max_mon_means, std_devs):
    """
    Function to calculate the 1, 2 and 3-level sigma thresholds for the month of maximum temperature for each grid cell
    Inputs:
    std_devs: xarray containing the standard deviations for the month of maximum temperature for each grid cell
    Outputs:
    thresholds
    """
    
    thresholds1 = (max_mon_means + std_devs).rename('threshold1')
    thresholds2 = (max_mon_means + 2*std_devs).rename('threshold2')
    thresholds3 = (max_mon_means + 3*std_devs).rename('threshold3')
    
    # merge into single dataset
    thresholds = xr.merge([thresholds1, thresholds2, thresholds3])
            
    return thresholds

def cold_extreme_thresholds(min_mon_means, std_devs):
    """
    Function to calculate the 1, 2 and 3-level sigma thresholds for the month of minimum temperature for each grid cell
    Inputs:
    std_devs: xarray containing the standard deviations for the month of minimum temperature for each grid cell
    Outputs:
    thresholds
    """
    
    thresholds1 = (min_mon_means - 0.25*std_devs).rename('threshold1')
    thresholds2 = (min_mon_means - 0.5*std_devs).rename('threshold2')
    thresholds3 = (min_mon_means - std_devs).rename('threshold3')
    
    # merge into single dataset
    thresholds = xr.merge([thresholds1, thresholds2, thresholds3])
            
    return thresholds