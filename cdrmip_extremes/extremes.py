"""
extremes.py
"""

import numpy as np
import xarray as xr

from collections import defaultdict

from xclim.core.calendar import percentile_doy
from xclim.indices import tx90p, warm_spell_duration_index, max_1day_precipitation_amount, max_n_day_precipitation_amount

from . import stats, utils


def monthly_extrema(ds, varname='tas'):
    """
    Month of maximum and minimum climatological-mean for each
    grid cell, from Dataset with 'time' dimension (typically
    piControl run). Returns ds unchanged if `varname` isn't present 

    Returns
    -------
    xr.Dataset with a 'month' variable, dims (..., 'extrema'), where
    extrema = ['max', 'min'].
    """
    if varname not in ds:
        return ds
    grouped = ds.groupby('time.month').mean(dim='time')
    extrema = xr.concat(
        [grouped[varname].idxmax(dim='month'), grouped[varname].idxmin(dim='month')],
        dim='extrema'
    ).assign_coords(extrema=['max', 'min']).rename('month')
    return extrema.to_dataset()


def extreme_month_stat(ds, months, stat='mean', varname='tas'):
    """
    Mean or std-dev of `varname` for the calendar month(s)
    given in `months`
    """
    da = ds[varname] if varname in ds else ds
    da_month = da.where(da.time.dt.month == months)
    if stat == 'mean':
        return da_month.mean(dim='time')
    elif stat == 'std':
        return da_month.std(dim='time')
    else:
        raise ValueError("stat must be 'mean' or 'std'")


def heat_extreme_thresholds(max_month_mean, max_month_std):
    """1, 2, 3-sigma thresholds above the climatological hottest-month mean."""
    return xr.merge([
        (max_month_mean + max_month_std).rename('threshold1'),
        (max_month_mean + 2 * max_month_std).rename('threshold2'),
        (max_month_mean + 3 * max_month_std).rename('threshold3'),
    ])


def cold_extreme_thresholds(min_month_mean, min_month_std):
    """
    -0.25, -0.5 and -1-sigma thresholds below the climatological coldest-month mean
    """
    return xr.merge([
        (min_month_mean - 0.25 * min_month_std).rename('threshold1'),
        (min_month_mean - 0.5 * min_month_std).rename('threshold2'),
        (min_month_mean - 1.0 * min_month_std).rename('threshold3'),
    ])


def select_extreme_month(ds, ext_months, ext_type_to_extrema=None, varname='tas'):
    """
    Restrict ds to the calendar month identified as 'heat' (hottest
    climatological month) or 'cold' (coldest), per grid cell, then take
    the annual mean over just that month for each year.

    ext_type_to_extrema : dict, optional
        Maps {'heat': <extrema label>, 'cold': <extrema label>} to
        whatever labels ext_months actually uses.
        Default is {'heat': 'max', 'cold': 'min'}

    Returns
    -------
    xr.DataArray with an 'extrema' dim labelled ['heat', 'cold']
    """
    if ext_type_to_extrema is None:
        ext_type_to_extrema = {'heat': 'max', 'cold': 'min'}

    da = ds[varname] if varname in ds else ds
    months = ext_months['month'] if 'month' in ext_months else ext_months

    dss = []
    for ext_type, extrema_label in ext_type_to_extrema.items():
        month = months.sel(extrema=extrema_label)
        dss.append(
            da.where(da.time.dt.month == month).groupby('time.year').mean(dim='time')
        )
    return xr.concat(dss, dim='extrema').assign_coords(extrema=list(ext_type_to_extrema.keys()))


def calculate_exceedances(monthly_temps, ext_thresholds, ext_type):
    """
    Fraction of years (as a percentage) in which a grid cell's extreme-
    month temperature exceeds each of three threshold levels.

    monthly_temps : xr.DataArray
        Output of select_extreme_month -- an 'extrema' dim labelled
        ['heat', 'cold'] and a 'year' dim (plus any other dims, e.g.
        gwl/branch, which ride along unaffected since only 'year' is
        reduced).
    ext_thresholds : xr.Dataset
        threshold1/2/3, from heat_extreme_thresholds/cold_extreme_thresholds.
    ext_type : {'heat', 'cold'}
    """
    n_years = monthly_temps.count(dim='year').mean().values
    da = monthly_temps.sel(extrema=ext_type)

    if ext_type == 'heat':
        compare = lambda thresh: da.where(da > thresh)
    elif ext_type == 'cold':
        compare = lambda thresh: da.where(da < thresh)
    else:
        raise ValueError("ext_type must be 'heat' or 'cold'")

    return xr.merge([
        ((compare(ext_thresholds.threshold1).count(dim='year') / n_years) * 100).rename('sigma1'),
        ((compare(ext_thresholds.threshold2).count(dim='year') / n_years) * 100).rename('sigma2'),
        ((compare(ext_thresholds.threshold3).count(dim='year') / n_years) * 100).rename('sigma3'),
    ])


def compute_multi_model_median(data_dict, dim='model'):
    """
    """
    stacked = xr.concat(
        list(data_dict.values()), dim=dim, compat='override', coords='minimal'
    )
    return stacked.median(dim=dim)


def calc_agreement(differences, threshold=0.75, drop_coords=None):
    """
    Multi-model sign agreement: fraction of models agreeing the change is
    positive, or fraction agreeing it's negative, at >= `threshold`.
 
    differences : dict {model: DataArray}
        Per-model difference field (e.g. ramp_down minus ramp_up).
    drop_coords : list of str, optional
        Coordinate names to drop from each model's data before
        concatenating along the new 'model' dimension.
    Returns
    -------
    xr.DataArray of 1 where >=`threshold` of models agree on sign
    (either direction), NaN elsewhere.
    """
    data = list(differences.values())
    if drop_coords:
        data = [
            d.drop_vars([c for c in drop_coords if c in d.coords])
            for d in data
        ]
    stacked = xr.concat(data, dim='model', compat='override', coords='minimal')
    agreement_pos = xr.where(stacked > 0, 1, 0).mean(dim='model')
    agreement_neg = xr.where(stacked < 0, 1, 0).mean(dim='model')
    agree = xr.where(agreement_pos >= threshold, 1, 0) + xr.where(agreement_neg >= threshold, 1, 0)
    return xr.where(agree != 0, 1, np.nan)
 
 
def calc_gwl_branch_differences(frequencies, exclude_keys=('Multi-Model Median', 'median'),
                                 drop_coords=None):
    """
    Ramp-down minus ramp-up exceedance-frequency difference for each
    model, plus the multi-model agreement on the sign of that difference.
 
    frequencies : dict {model: Dataset/DataArray with a 'branch' dim}
    exclude_keys : keys to skip when computing agreement 
    drop_coords : list of str, optional
        Passed through to calc_agreement. Pass
        ['gwl'] when frequencies came from gwl_matching rather than
        crossing_years.
    """
    differences = {
        model: data.sel(branch='ramp_down') - data.sel(branch='ramp_up')
        for model, data in frequencies.items()
    }
    agreement = calc_agreement(
        {m: d for m, d in differences.items() if m not in exclude_keys},
        drop_coords=drop_coords,
    )
    return differences, agreement

# -------------------------------------------------------------------------
# ETCCDI metrics
# -------------------------------------------------------------------------

def calc_txx(ds: xr.Dataset) -> xr.Dataset:
    if "tasmax" in ds:  # only operate if tasmax exists
        txx = ds["tasmax"].groupby("time.year").max(dim="time")
        return txx.to_dataset(name="txx")
    return ds  # return unchanged if no tasmax

def calc_txx_anomaly(txx_ds, reference_period):
    # txx has an integer 'year' dim, so reference_period is slice(1850, 1900)
    return utils.calc_anomaly(txx_ds, txx_ds, reference_period, time_dim="year")

def calc_tx90p(ds, ref_period):
    if "tasmax" not in ds:
        return ds
 
    tasmax = ds["tasmax"]
    tasmax = tasmax.sortby("time").drop_duplicates("time")
    tasmax = tasmax.convert_calendar("noleap", align_on="date")
    if not tasmax.attrs.get("units"):
        tasmax = tasmax.assign_attrs(units="K")

    # --- THRESHOLD ---------------------------------------------------------
    lo, hi = ref_period
    mask = (tasmax["time.year"] >= lo) & (tasmax["time.year"] <= hi)
    ref_ds = (
        tasmax.isel(time=mask)
        .chunk({"time": -1, "lat": 45, "lon": 45})
    )
    thresh_doy = percentile_doy(ref_ds, per=90).sel(percentiles=90).compute()
 
    # --- INDEX -------------------------------------------------------------
    # tx90p compares each day to its day-of-year threshold: a per-timestep op
    # that works on time-CHUNKED data. It never needs time=-1. Keep modest
    # chunks so nothing gets rechunked into a giant contiguous block.
    tasmax = tasmax.chunk({"time": 365, "lat": 45, "lon": 45})
    tx90 = tx90p(tasmax, thresh_doy).groupby("time.year").mean(dim="time")
    return tx90.to_dataset(name="tx90p")

def calc_rxnday(ds: xr.Dataset, n: int = 1) -> xr.Dataset:
    """
    Calculate the ETCCDI RXnday index: annual maximum of the
    running n-day precipitation total (n=1 -> Rx1day, n=5 -> Rx5day).
    """
    pr = ds['pr']
    if n > 1:
        pr = pr.rolling(time=n, min_periods=n).sum()
    return pr.groupby('time.year').max(dim='time').to_dataset(name=f'rx{n}day')