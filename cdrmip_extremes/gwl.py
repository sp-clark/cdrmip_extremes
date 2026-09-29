"""
gwl.py

This module contains the functions necessary to determine the years in which specified global warming
levels (GWLs) are exceeded for different experiments and to extract the associated 'gwl periods' (i.e.
the 21-year period centred on the year of GWL crossing) for given datasets
"""

from collections import defaultdict

import xarray as xr
import numpy as np
from . import stats

# ---------------------------------------------------------------------------
# GWL crossing years
# ---------------------------------------------------------------------------

def find_crossing_years(gsat_da, window, gwls, time_dim='year', overshoot=True):
    """
    Find the year(s) at which a rolling-mean GSAT timeseries crosses each
    of `gwls`.
 
    overshoot=True: for each gwl, returns the ramp_up crossing (last time
        warming first reaches gwl on the way up to peak) and ramp_down
        crossing (first time it drops back through gwl after peak).
    overshoot=False: returns a single crossing year per gwl (assumes
        monotonic warming).
 
    Returns
    -------
    xr.DataArray
        dims ('gwl', 'branch') if overshoot, else dims ('gwl',).
        Missing crossings (gwl never reached) are NaN rather than raising.
    """
    rolling = stats._rolling_mean(gsat_da, window, time_dim)
 
    if overshoot:
        peak_year = stats.find_peak_year(rolling)
        pre_peak = rolling.sel(year=slice(None, peak_year))
        post_peak = rolling.sel(year=slice(peak_year + 1, None))
 
        ramp_up, ramp_down = [], []
        for gwl in gwls:
            pre_match = pre_peak.where(pre_peak >= gwl).dropna(dim='year')
            post_match = post_peak.where(post_peak <= gwl).dropna(dim='year')
            ramp_up.append(int(pre_match.year[0]) if pre_match.year.size else np.nan)
            ramp_down.append(int(post_match.year[0]) if post_match.year.size else np.nan)
 
        da = xr.DataArray(
            np.array([ramp_up, ramp_down]).T,
            coords={'gwl': gwls, 'branch': ['ramp_up', 'ramp_down']},
            dims=['gwl', 'branch'],
        )
    else:
        crossings = []
        for gwl in gwls:
            match = rolling.where(rolling >= gwl).dropna(dim='year')
            crossings.append(int(match.year[0]) if match.year.size else np.nan)
 
        da = xr.DataArray(crossings, coords={'gwl': gwls}, dims=['gwl'])
 
    return da
# ---------------------------------------------------------------------------
# Extract GWL period
# ---------------------------------------------------------------------------

def extract_gwl_period(ds, gwl_years, window, time_dim='year'):
    """
    Extract multi-year windows of a timeseries centred on GWL crossing years.

    Windows keep their absolute calendar years, so the output's 'year' axis
    is the union of all windows (mostly NaN per GWL). GWLs that are never
    crossed come back as all-NaN.

    Parameters
    ----------
    ds : xarray.DataArray
        Yearly time series with a `time_dim` dimension.
    gwl_years : xarray.DataArray
        Output of find_crossing_years -- dims ('gwl', 'branch') or ('gwl',).
    window : int
        Window length, centred on the crossing year.

    Returns
    -------
    xarray.DataArray with dims ('branch', 'gwl', 'year', ...) or
    ('gwl', 'year', ...).
    """
    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')

    half = (window - 1) // 2

    def extract_one(year):
        if np.isnan(year):
            return ds.isel(year=slice(0, 0))  # empty; filled with NaN by the outer join
        year = int(year)
        return ds.sel(year=slice(year - half, year + half))

    def extract_windows(years):
        return xr.concat([extract_one(y) for y in years.values], dim='gwl', join='outer')

    if 'branch' in gwl_years.dims:
        out = xr.concat(
            [extract_windows(gwl_years.sel(branch=b)) for b in gwl_years.branch.values],
            dim='branch', join='outer',
        )
        return out.assign_coords(branch=gwl_years.branch.values, gwl=gwl_years.gwl.values)
    return extract_windows(gwl_years).assign_coords(gwl=gwl_years.gwl.values)

# ---------------------------------------------------------------------------
# GWL branch differences
# ---------------------------------------------------------------------------
def gwl_branch_difference(gwl_period_da, dim='year', percent=False):
    """
    Difference between the ramp_down and ramp_up branches at the same GWL,
    time-averaged over `dim` first if present.
    """
    if 'branch' not in gwl_period_da.dims:
        raise ValueError(
            "gwl_period_da has no 'branch' dimension -- was it computed "
            "with overshoot=True? Branch-difference is only meaningful "
            "for overshoot experiments with a ramp_up/ramp_down pair."
        )
    if dim in gwl_period_da.dims:
        gwl_period_da = gwl_period_da.mean(dim=dim, skipna=True)
    down = gwl_period_da.sel(branch='ramp_down')
    up = gwl_period_da.sel(branch='ramp_up')
    diff = down - up
    return 100 * diff / up if percent else diff
# ---------------------------------------------------------------------------
# GWL matching years
# ---------------------------------------------------------------------------
def find_matching_gwl_year(gsat_da, window, match_slice, time_dim='year', search_slice=None):
    """
    Compute the mean rolling GSAT over `match_slice` (e.g. the final ~20
    years of a run, or a late-century window), then find the earliest year
    within `search_slice` at which the rolling timeseries first reached
    that same level.
 
    search_slice : slice, optional
        Restricts where to look for the match. Defaults to everything
        before peak warming.
        Pass slice(None, None) to search the whole series
 
    Returns
    -------
    (gwl, matching_year) : tuple[float, int]
    """
    rolling = stats._rolling_mean(gsat_da, window, time_dim)
 
    end_gwl = np.round(float(rolling.sel(year=match_slice).mean(dim='year').values), 2)

    end_mid_year = int((match_slice.start + (match_slice.stop)) / 2)
 
    if search_slice is None:
        peak_year = stats.find_peak_year(rolling)
        search_da = rolling.sel(year=slice(None, peak_year))
    else:
        search_da = rolling.sel(year=search_slice)
 
    match = search_da.where(search_da >= end_gwl).dropna(dim='year')
    if match.year.size == 0:
        raise ValueError(f"No year found where rolling GSAT reaches {end_gwl}")
 
    matching_year = int(match.year[0])

    da = xr.DataArray(
        np.array([[matching_year],[end_mid_year]]).T,
        coords={'gwl':[end_gwl],'branch':['ramp_up','ramp_down']},
        dims=['gwl','branch'])

    return da


def gwl_matched_difference(anom_da, post_slice, matching_year,
                                   half_window=None, time_dim='year'):
    """
    Difference between the mean anomaly over `post_slice` and the mean
    anomaly over an equal-length window centred on `matching_year`.
    Window length is inferred from `post_slice` unless `half_window`
    is given explicitly.
    """
    post_da = anom_da.sel({time_dim: post_slice})
    if half_window is None:
        window = post_da[time_dim].size          # actual years selected
        half_window = (window - 1) // 2
    pre_mean = anom_da.sel(
        {time_dim: slice(matching_year - half_window, matching_year + half_window)}
    ).mean(time_dim)
    return post_da.mean(time_dim) - pre_mean