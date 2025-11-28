import xarray as xr
import numpy as np
import cftime


def monthly_extrema(da):
    """
    Returns the months of maximum and minimum temperature for each grid cell
    """
    grouped = da.groupby('time.month').mean('time')
    extrema = xr.concat(
        [grouped.idxmax('month'), grouped.idxmin('month')],
        dim='extrema'
    ).assign_coords({'extrema':['max','min']}).rename({'tas':'month'})
    return extrema

def monthly_extrema_dt(ds:xr.Dataset) -> xr.Dataset:
    if 'tas' in ds:
        return monthly_extrema(ds)
    else:
        return ds

def select_extreme_month(ds,ext_months):
    dss = []
    for ext_type in ['heat','cold']:
        dss.append(
            ds.where(
                ds.time.dt.month==ext_months.sel(extrema=ext_type).month
            ).groupby('time.year').mean(dim='time')
        )
    return xr.concat(dss,dim='extrema')
    

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


def calculate_exceedances(monthly_temps,ext_thresholds,ext_type):

    years =monthly_temps.count(dim='year').mean().values
    
    # Now find which grid-cells feature extreme events for each year
    if ext_type == 'heat':
        monthly_temps = monthly_temps.sel(extrema=ext_type)
        exceedances1 = monthly_temps.where(monthly_temps > ext_thresholds.threshold1)
        exceedances2 = monthly_temps.where(monthly_temps > ext_thresholds.threshold2)
        exceedances3 = monthly_temps.where(monthly_temps > ext_thresholds.threshold3)
    elif ext_type == 'cold':
        exceedances1 = monthly_temps.where(monthly_temps < ext_thresholds.threshold1)
        exceedances2 = monthly_temps.where(monthly_temps < ext_thresholds.threshold2)
        exceedances3 = monthly_temps.where(monthly_temps < ext_thresholds.threshold3)
    
    # these three xarrays thus describe those grid-cells where in a given year an extreme heat event is recorded
    # if no event is recorded in that year, the grid-cell entry will be nan
    
    # We are seeking to calculate the proportion of years for each period that feature extreme heat events
    # thus we are only concerned really with non-nan values, and can calculate how many such values occur for each grid-cell
    # using xr.count()
    
    years_exceeding1 = ((exceedances1.count(dim='year')/years)*100).rename('sigma1') # remembering that we are seeking percentages
    years_exceeding2 = ((exceedances2.count(dim='year')/years)*100).rename('sigma2') # remembering that we are seeking percentages
    years_exceeding3 = ((exceedances3.count(dim='year')/years)*100).rename('sigma3') # remembering that we are seeking percentages
    
    # merge these into a single dataset for simplicity
    exceedance_frequencies = xr.merge([years_exceeding1, years_exceeding2, years_exceeding3])
    
    return exceedance_frequencies


def calc_agreement(differences):
    differences_all = xr.concat(
        list(differences.values()),
        dim='model',
        compat='override',
        coords='minimal'
    )
    
    # find areas where there are positive or negative changes
    difference_pos = xr.where(differences_all > 0,1,0)
    difference_neg = xr.where(differences_all < 0,1,0)
    
    # find fraction of models agreeing on sign being positive by taking mean across models
    agreement_pos = difference_pos.mean(dim='model')
    # find fraction agreeing on sign being negative
    agreement_neg = difference_neg.mean(dim='model')
    
    # areas where 6/8 models agree sign is positive or where 6/8 agree that sign is negative
    agreement_all = xr.where(agreement_pos>=0.75,1,0) + xr.where(agreement_neg>=0.75,1,0)
    agreement = xr.where(agreement_all !=0,1,np.nan)

    return agreement
def calc_gwl_differences(frequencies):
    differences = {}
    for model in frequencies.keys():
        ramp_up = frequencies[model].sel(branch='ramp_up')
        ramp_down = frequencies[model].sel(branch='ramp_down')
        differences[model] = ramp_down - ramp_up
    agreement = calc_agreement({model:difference for model, difference in differences.items() if model != 'Multi-Model Median'})
    return differences, agreement