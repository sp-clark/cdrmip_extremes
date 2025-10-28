import xarray as xr
import numpy as np
import cftime


def monthly_extrema(da):
    grouped = da.groupby('time.month').mean('time')
    return grouped.idxmax('month'), grouped.idxmin('month')
    
def max_month_mean(max_months,tas_data):
    """
    Returns a gridded xarray specifying the mean tas value for the month of maximum temperature in each grid cell for a given simulation
    """
    return tas_data.where(tas_data.time.dt.month==max_months).mean(dim='time') # Note that taking mean preserves the monthly tas value as xr.mean() ignores NaN values

def min_month_mean(min_months,tas_data):
    """
    Returns a gridded xarray specifying the mean tas value for the month of minimum temperature in each grid cell for a given simulation
    """
    return tas_data.where(tas_data.time.dt.month==min_months).mean(dim='time')

def max_month_std_dev(max_months,tas_data):
    """
    Returns a gridded xarray specifying the standard deviation values for the month of maximum temperature in each grid cell for a given simulation
    """
    return tas_data.where(tas_data.time.dt.month==max_months).std(dim='time')

def min_month_std_dev(min_months,tas_data):
    """
    Returns a gridded xarray specifying the standard deviation values for the month of minimum temperature in each grid cell for a given simulation
    """
    return tas_data.where(tas_data.time.dt.month==min_months).std(dim='time')

def heat_extreme_thresholds(tas_data, max_mon_means, std_devs):
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

def cold_extreme_thresholds(tas_data, min_mon_means, std_devs):
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