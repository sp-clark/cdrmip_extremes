"""
extremes.py
"""

import numpy as np
import xarray as xr
import cftime

from collections import defaultdict

from xclim.core.calendar import percentile_doy
from xclim.indices import tx90p, tn10p, warm_spell_duration_index, max_1day_precipitation_amount, max_n_day_precipitation_amount

from . import stats, utils

# ---------------------------------------------------------------------------
# hottest/coldest month analysis
# ---------------------------------------------------------------------------

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
    months = ext_months['month'] if isinstance(ext_months, xr.Dataset) else ext_months

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

# -------------------------------------------------------------------------
# Mean, Median or Std. of extreme month temperatures
# -------------------------------------------------------------------------

def ext_month_stat_diff(ext_mon_data, extrema, method="mean"):
    """
    Ramp-down minus ramp-up `method` ('mean', 'std', 'max', ...) over years of
    the extreme-month data, for each model/expt in {model: {expt: ds}}.
    """
    def diff(ds):
        ds = ds.sel(extrema=extrema)
        stat = lambda branch: getattr(ds.sel(branch=branch), method)(dim="year")
        return stat("ramp_down") - stat("ramp_up")
    return utils.map_nested(diff, ext_mon_data)


# -------------------------------------------------------------------------
# ETCCDI metrics
# -------------------------------------------------------------------------
def calc_txx(ds):
    if "tasmax" not in ds:
        return None
    txx = ds["tasmax"].groupby("time.year").max("time")
    return txx.chunk({"year": -1}).to_dataset(name="txx")

def calc_tnn(ds):
    if "tasmin" not in ds:
        return None
    tnn = ds["tasmin"].groupby("time.year").min("time")
    return tnn.chunk({"year": -1}).to_dataset(name="tnn")

def calc_rxnday(ds, n=1):
    """ETCCDI RXnday: annual max of the running n-day precip total."""
    if "pr" not in ds:
        return None
    pr = ds["pr"]
    if n > 1:
        pr = pr.rolling(time=n, min_periods=n).sum()
    return pr.groupby("time.year").max("time").chunk({"year": -1}).to_dataset(name=f"rx{n}day")

def period_anomaly(index_ds, ref_period):
    """Anomaly vs the index's own mean over ref_period=(lo, hi).
    Returns None (member dropped) if the series doesn't cover it."""
    lo, hi = ref_period
    yrs = index_ds["year"]
    if int(yrs.min()) > lo or int(yrs.max()) < hi:
        return None
    return index_ds - index_ds.sel(year=slice(lo, hi)).mean("year")

def picontrol_anomaly(index_tree, pi_ref):
    """{model: {expt: {member: ds}}} minus {model: {member: ref}}.
    Members without a piControl reference are dropped."""
    return {
        model: {
            expt: {mem: ds - pi_ref[model][mem]
                   for mem, ds in mems.items() if mem in pi_ref.get(model, {})}
            for expt, mems in expts.items()
        }
        for model, expts in index_tree.items()
    }


def _prep_temp(ds, var):
    da = ds[var].sortby("time").drop_duplicates("time")
    da = da.convert_calendar("noleap", align_on="year" if da.time.dt.calendar == "360_day" else "date")
    da["time"] = [cftime.DatetimeNoLeap(t.year, t.month, t.day, t.hour) for t in da.time.values]
    if not da.attrs.get("units"):
        da = da.assign_attrs(units="K")
    return da

def _doy_threshold(ds, var, per, ref_period):
    """Day-of-year `per`th percentile of `var` over ref_period=(lo, hi); None bounds -> series ends."""
    if var not in ds:
        return None
    da = _prep_temp(ds, var)
    years = da["time.year"]
    yr_lo, yr_hi = int(years.min()), int(years.max())
    lo = yr_lo if ref_period[0] is None else ref_period[0]
    hi = yr_hi if ref_period[1] is None else ref_period[1]
    if yr_lo > lo or yr_hi < hi:
        return None
    ref = da.isel(time=(years >= lo) & (years <= hi)).chunk({"time": -1, "lat": 10, "lon": 10})
    return percentile_doy(ref, per=per).sel(percentiles=per).compute()

def tx90p_threshold(ds, ref_period=(None, None)):
    return _doy_threshold(ds, "tasmax", 90, ref_period)

def tn10p_threshold(ds, ref_period=(None, None)):
    return _doy_threshold(ds, "tasmin", 10, ref_period)

def calc_tx90p(ds, thresh_doy):
    """Annual TX90p of ds against a precomputed day-of-year threshold."""
    if "tasmax" not in ds or thresh_doy is None:
        return None
    tasmax = _prep_temp(ds, "tasmax").chunk({"time": 365, "lat": 45, "lon": 45})
    tx90 = tx90p(tasmax, thresh_doy.chunk({"lat": 45, "lon": 45})).groupby("time.year").mean(dim="time")
    return tx90.chunk({"year": -1}).to_dataset(name="tx90p")

def calc_tn10p(ds, thresh_doy):
    """Annual TN10p of ds against a precomputed day-of-year threshold."""
    if "tasmin" not in ds or thresh_doy is None:
        return None
    tasmin = _prep_temp(ds, "tasmin").chunk({"time": 365, "lat": 45, "lon": 45})
    tn10 = tn10p(tasmin, thresh_doy.chunk({"lat": 45, "lon": 45})).groupby("time.year").mean(dim="time")
    return tn10.chunk({"year": -1}).to_dataset(name="tn10p")

