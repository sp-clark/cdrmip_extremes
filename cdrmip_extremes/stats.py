"""
stats.py
"""
 
import numpy as np
import xarray as xr
from collections import defaultdict

from .utils import _pop_coords, _restore_reduced_coords
 
 
def _rolling_mean(da, window, time_dim):
    """Annual-mean (if needed) + centred rolling mean."""
    if time_dim != 'year':
        da = da.groupby('time.year').mean(dim='time')
        time_dim = 'year'
    return da.rolling({time_dim: window}, center=True, min_periods=1).mean()
 
 
def find_peak_year(rolling_da, time_dim='year'):
    return int(rolling_da.idxmax(dim=time_dim))

def peak_warming(ds, time_dim='year',round=True):
    """
    Find the peak value and the year it occurs in a (rolling)
    timeseries.
 
    Returns
    -------
    (peak, peak_year) : tuple[float, int]
    """
    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')
        time_dim = 'year'
 
    peak_year = find_peak_year(ds, time_dim=time_dim)
    if round:
        peak = np.round(float(ds.sel({time_dim: peak_year}).values), 2)
    else: 
        peak = ds.sel({time_dim: peak_year})
    return peak, peak_year

def branch_difference(da, turn=140):
    """
    """
    pre = da.sel(year=slice(0, turn))
    post = da.sel(year=slice(turn, 2 * turn)).assign_coords(year=pre.year[::-1])
    return post - pre


# ---------------------------------------------------------------------------
# model agreement
# ---------------------------------------------------------------------------


def calc_agreement(differences, threshold=0.75, expt=None, drop_coords=None):
    """
    ...
    drop_coords : list, optional
        Coords to drop before stacking models (e.g. ['gwl'] when each model
        matched a slightly different GWL). The median is re-attached afterwards.
    """
    if expt is not None:
        differences = {m: d[expt] for m, d in differences.items() if expt in d}

    dropped = {}
    fields = [_pop_coords(d, drop_coords, dropped) for d in differences.values()]
    stacked = xr.concat(fields, dim="model", compat="override", coords="minimal")
    agreement_pos = (stacked > 0).mean(dim="model")
    agreement_neg = (stacked < 0).mean(dim="model")
    agreement = xr.where((agreement_pos >= threshold) | (agreement_neg >= threshold), 1, np.nan)
    return _restore_reduced_coords(agreement, dropped, "median")



# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------
def regress_pattern_with_r2(X,Y, time_dim="time"):
    """
    Perform regression of tas against gmst for xarray objects.
    Returns slope (lat, lon) and r^2 (lat, lon).
    """

    # Regression slope = cov(X,Y) / var(X)
    cov = xr.cov(X, Y, dim=time_dim)
    var = X.var(dim=time_dim)
    slope = cov / var

    # Predicted tas from regression
    Y_hat = slope * X

    # Residuals
    residuals = Y - Y_hat

    # Sum of squares
    ss_res = (residuals**2).sum(dim=time_dim)
    ss_tot = ((Y - Y.mean(dim=time_dim))**2).sum(dim=time_dim)

    r2 = 1 - ss_res / ss_tot

    return slope, r2

def compute_pre_post_patterns(data, gsat_rolling, peaks, start=None, end=None):
    patterns = {}

    for model, expt_dt in data.items():
        for expt, member_dt in expt_dt.items():
            for member, ds in member_dt.items():

                peak_year = peaks[model][expt][member]['peak_year']
        
                if start == 'peak':
                    start = peak_year
                if end == 'peak':
                    end = peak_year
        
                Y = ds.sel(year=slice(start, end))
                X = gsat_rolling[model][expt][member].sel(year=slice(start, end))

                slope, r2 = regress_pattern_with_r2(X, Y, time_dim='year')
                patterns[model] = {'slope':slope,'r2':r2}

    return patterns