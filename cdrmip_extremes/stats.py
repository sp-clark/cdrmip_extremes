"""
stats.py
"""
 
import numpy as np
import xarray as xr
 
 
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
    rolling = _rolling_mean(gsat_da, window, time_dim)
 
    if overshoot:
        peak_year = find_peak_year(rolling)
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
    rolling = _rolling_mean(gsat_da, window, time_dim)
 
    end_gwl = np.round(float(rolling.sel(year=match_slice).mean(dim='year').values), 2)

    end_mid_year = int((match_slice.start + (match_slice.stop)) / 2)
 
    if search_slice is None:
        peak_year = find_peak_year(rolling)
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
 

def gwl_matched_anomaly_difference(anom_da, post_slice, matching_year,
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

def extract_gwl_period(ds, gwl_years, window, time_dim='year', coord='year'):
    """
    Extract multi-year windows of a timeseries centred on GWL crossing
    years.
 
    coord : {'year', 'lag'}, default 'year'
        'year': windows keep their original absolute calendar year as
            the coordinate. Note that since different models/GWLs/
            branches cross at different years, windows won't share a
            common axis -- concatenating windows across models on this
            coordinate produces a sparse, mostly-NaN result unless you
            re-align by position afterwards. Windows with a missing
            (NaN) crossing year get an out-of-range sentinel year label
            (so they don't collide with real years) rather than being
            dropped.
        'lag': windows are expressed as an offset from the crossing year
            (-half_window...+half_window), making them directly
            comparable across models/GWLs/branches regardless of which
            actual years they cover. Missing crossings get an all-NaN
            window on this same axis.
 
    Parameters
    ----------
    ds : xarray.DataArray
        Yearly time series with a `time_dim` dimension.
    gwl_years : xarray.DataArray
        Output of find_crossing_years -- dims ('gwl', 'branch') (or just
        ('gwl',) if produced with overshoot=False).
    window : int
        Length of the window to extract, e.g. 21. Centred on the crossing
        year; half_window = (window - 1) // 2 years either side.
 
    Returns
    -------
    xarray.DataArray with dims ('branch', 'gwl', 'year'|'lag', ...) -- or
    ('gwl', 'year'|'lag', ...) if gwl_years has no 'branch' dimension.
    """
    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')
 
    half = (window - 1) // 2
    gwls = gwl_years.gwl.values
    _SENTINEL_BASE = -1_000_000  # clearly out-of-range, used only for missing crossings
 
    def extract_one(year):
        if np.isnan(year):
            template = ds.isel(year=0, drop=True)
            nan_window = xr.full_like(template, np.nan).expand_dims(year=window)
            if coord == 'lag':
                return nan_window.assign_coords(year=np.arange(-half, half + 1)).rename(year='lag')
            else:
                return nan_window.assign_coords(year=_SENTINEL_BASE + np.arange(window))
 
        year = int(year)
        target_years = np.arange(year - half, year + half + 1)
        window_ds = ds.reindex(year=target_years)
        if coord == 'lag':
            window_ds = window_ds.assign_coords(year=window_ds['year'] - year).rename(year='lag')
        return window_ds
 
    def extract_windows(years):
        periods = [extract_one(year) for year in years]
        return xr.concat(periods, dim='gwl').assign_coords(gwl=gwls)
 
    if 'branch' in gwl_years.dims:
        branches = gwl_years.branch.values
        results = [
            extract_windows(gwl_years.sel(branch=branch).values)
            for branch in branches
        ]
        return xr.concat(results, dim='branch').assign_coords(branch=branches)
    else:
        return extract_windows(gwl_years.values)

def gwl_branch_difference(gwl_period_da, dim='year', percent=False):
    """
    Time-mean difference between the ramp_down and ramp_up branches of a
    GWL-period extraction (output of extract_gwl_period), for each gwl --
    i.e. a hysteresis/reversibility metric: does the climate look the same
    at a given warming level on the way down as it did on the way up.
 
    Parameters
    ----------
    gwl_period_da : xarray.DataArray
        Output of extract_gwl_period, with dims including 'branch' and
        `dim` ('year' or 'lag', matching whatever `coord` was used).
    percent : bool, default False
        If True, return the percentage difference relative to ramp_down,
        i.e. 100 * (ramp_down - ramp_up) / ramp_down, instead of the raw
        difference. Points where ramp_down is 0 return inf/NaN.
 
    Returns
    -------
    xarray.DataArray with dims ('gwl', ...) -- 'branch' and `dim` are
    consumed (averaged over the time-window dim, differenced over branch).
    """
    if 'branch' not in gwl_period_da.dims:
        raise ValueError(
            "gwl_period_da has no 'branch' dimension -- was it computed "
            "with overshoot=True? Branch-difference is only meaningful "
            "for overshoot experiments with a ramp_up/ramp_down pair."
        )
    branch_mean = gwl_period_da.mean(dim=dim, skipna=True)
    down = branch_mean.sel(branch='ramp_down')
    up = branch_mean.sel(branch='ramp_up')
    diff = down - branch_mean.sel(branch='ramp_up')
    if percent:
        return 100 * diff / up
    return diff

def _pop_coords(ds, drop_coords, store):
    """Drop `drop_coords` from ds, stashing their values in `store` so the
    reduction can be applied to them too (instead of losing them)."""
    if drop_coords:
        for c in drop_coords:
            if c in ds.coords:
                store.setdefault(c, []).append(np.atleast_1d(ds[c].values))
                ds = ds.drop_vars(c)
    return ds
 
 
def _restore_reduced_coords(reduced, store, reduction):
    """Re-attach dropped coords to the reduced result, applying the same
    reduction across members/models (e.g. median gwl level)."""
    np_reduce = getattr(np, f"nan{reduction}", None) or getattr(np, reduction)
    for c, vals in store.items():
        try:
            summary = np_reduce(np.stack(vals), axis=0)
        except (ValueError, TypeError):
            continue  # inconsistent shapes / non-numeric -> leave dropped
        if c in reduced.dims and summary.shape == (reduced.sizes[c],):
            reduced = reduced.assign_coords({c: summary})
    return reduced
 
 
def reduce_members(member_dt, reduction="median", drop_coords=None):
    """
    """
    datasets = []
    dropped = {}
    for member, ds in member_dt.items():
        ds = _pop_coords(ds, drop_coords, dropped)
        datasets.append(ds.expand_dims(member=[member]))
 
    combined = xr.concat(datasets, dim="member")
    reduced = getattr(combined, reduction)(dim="member")
    return _restore_reduced_coords(reduced, dropped, reduction)
 
def reduce_models(nested_dict, reduction="median", expts=None, drop_coords=None):
    """
    """
    output = nested_dict.copy()
 
    if expts is None:
        first_model = next(iter(nested_dict))
        expts = nested_dict[first_model].keys()
 
    output[f"multi_model_{reduction}"] = {}
 
    for expt in expts:
        datasets = []
        dropped = {}
        for model, expt_dt in nested_dict.items():
            if expt not in expt_dt:
                continue
            ds = _pop_coords(expt_dt[expt], drop_coords, dropped)
            datasets.append(ds.expand_dims(model=[model]))
 
        reduced = getattr(xr.concat(datasets, dim="model"), reduction)(dim="model")
        output[f"multi_model_{reduction}"][expt] = _restore_reduced_coords(
            reduced, dropped, reduction
        )
 
    return output
 
def _is_leaf(x):
    return isinstance(x, (xr.DataArray, xr.Dataset))
 
 
def _reduce_recursive(model_dict, reduction, drop_coords):
    sample_value = next(iter(model_dict.values()))
 
    if _is_leaf(sample_value):
        datasets = []
        dropped = {}
        for model, ds in model_dict.items():
            ds = _pop_coords(ds, drop_coords, dropped)
            datasets.append(ds.expand_dims(model=[model]))
        reduced = getattr(xr.concat(datasets, dim="model"), reduction)(dim="model")
        return _restore_reduced_coords(reduced, dropped, reduction)
 
    # sample_value is dict-like (expt, var, member, ...) -- recurse,
    # using the union of keys across models in case some are missing
    # a key (e.g. a model that didn't run a given experiment)
    all_keys = []
    for v in model_dict.values():
        for k in v.keys():
            if k not in all_keys:
                all_keys.append(k)
 
    result = {}
    for key in all_keys:
        sub_dict = {model: v[key] for model, v in model_dict.items() if key in v}
        if sub_dict:
            result[key] = _reduce_recursive(sub_dict, reduction, drop_coords)
    return result
 
 
def reduce_models_recursive(nested_dict, reduction="median", drop_coords=None):
    """
    """
    output = dict(nested_dict)
    output[f"multi_model_{reduction}"] = _reduce_recursive(nested_dict, reduction, drop_coords)
    return output

def calc_agreement(differences, threshold=0.75):
    """
    Multi-model sign agreement: fraction of models agreeing the change is
    positive, or fraction agreeing it's negative, at >= `threshold`.

    differences : dict {model: DataArray}
        Per-model difference field (e.g. ramp_down minus ramp_up).

    Returns
    -------
    xr.DataArray of 1 where >=`threshold` of models agree on sign
    (either direction), NaN elsewhere.
    """
    stacked = xr.concat(
        list(differences.values()), dim='model', compat='override', coords='minimal'
    )
    agreement_pos = xr.where(stacked > 0, 1, 0).mean(dim='model')
    agreement_neg = xr.where(stacked < 0, 1, 0).mean(dim='model')
    agree = xr.where(agreement_pos >= threshold, 1, 0) + xr.where(agreement_neg >= threshold, 1, 0)
    return xr.where(agree != 0, 1, np.nan)

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