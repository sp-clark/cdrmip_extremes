import xarray as xr
import numpy as np
import cftime
from collections import defaultdict

from cdrmip_extremes import utils

def _extract_window(ds, year, window, time_dim='year'):
    """Extract a single window of `ds` centred on `year`."""
    start = year - (window - 1) // 2
    end = year + (window - 1) // 2
    return ds.sel({time_dim: slice(start, end)})

def extract_gwl_period_scenariomip(ds, gwl_years_ds, window, time_dim='year'):
    """
    Extract multi-year windows of a timeseries centred around
    global warming level (GWL) crossing years, for the
    expt-based gwl_years format (single model, single member).

    Parameters
    ----------
    ds : xarray.DataArray
        Yearly time series (e.g., temperature, precipitation, etc.)
        with a 'year' dimension, for a single model and member.
    gwl_years : xarray.Dataset
        Dataset with coordinate 'expt', and data variables 'gwl',
        'pre_gwl_year', 'post_gwl_year', already subset to the
        model and member corresponding to `ds`.
    window : int
        Length of the window to extract, typically 21 years.
        The window is centred on the GWL crossing year.
    time_dim : str
        Name of the time dimension in `ds` if not already 'year'.

    Returns
    -------
    xarray.DataArray
        Concatenated DataArray containing the values of `ds` for each
        expt, gwl and period ('pre'/'post') within the specified
        time window.
        Dimensions: ['period', 'expt', 'year', ...].
    """
    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')

    gwl_val = gwl_years_ds['gwl'].item()
    if np.isnan(gwl_val):
        return None

    pre_year = gwl_years_ds['pre_gwl_year'].item()
    post_year = gwl_years_ds['post_gwl_year'].item()

    pre_window = _extract_window(ds, pre_year, window, time_dim='year')
    post_window = _extract_window(ds, post_year, window, time_dim='year')

    period_da = xr.concat([pre_window, post_window], dim='period')
    period_da = period_da.assign_coords(period=['pre', 'post'])
    period_da = period_da.assign_coords(gwl=gwl_val)

    return period_da

def calc_agreement(differences, as_datatree=False):
    """
    differences : nested dict { model: { expt: { var: ds } } }
    Returns nested dict { expt: { var: agreement_ds } } or xr.DataTree
    """
    regrouped = defaultdict(lambda: defaultdict(dict))
    for model, expt_dict in differences.items():
        for expt, var_dict in expt_dict.items():
            for var, ds in var_dict.items():
                regrouped[expt][var][model] = ds

    agreement = defaultdict(dict)
    for expt, var_dict in regrouped.items():
        for var, model_dict in var_dict.items():
            differences_all = xr.concat(
                list(model_dict.values()),
                dim='model',
                compat='override',
                coords='minimal'
            )

            difference_pos = xr.where(differences_all > 0, 1, 0)
            difference_neg = xr.where(differences_all < 0, 1, 0)

            agreement_pos = difference_pos.mean(dim='model')
            agreement_neg = difference_neg.mean(dim='model')

            agreement_all = xr.where(agreement_pos >= 0.75, 1, 0) + xr.where(agreement_neg >= 0.75, 1, 0)
            agreement[expt][var] = xr.where(agreement_all != 0, 1, np.nan)

    if as_datatree:
        tree_dict = {
            f"/{expt}/{var}": ds
            for expt, var_dict in agreement.items()
            for var, ds in var_dict.items()
        }
        return xr.DataTree.from_dict(tree_dict)

    return utils.defaultdict_to_dict(agreement)


def calc_gwl_differences(ext_freq_data, pre_name='pre', post_name='post',
                          period_coord='period', as_datatree=False):
    differences = defaultdict(lambda: defaultdict(dict))
    for model, expt_dt in ext_freq_data.items():
        for expt, var_dict in expt_dt.items():
            for var, leaf in var_dict.items():
                # accept either a DataTree node (.ds) or a plain Dataset
                ds = leaf.ds if hasattr(leaf, 'ds') else leaf
                pre_gwl = ds.sel({period_coord: pre_name})
                post_gwl = ds.sel({period_coord: post_name})
                differences[model][expt][var] = post_gwl - pre_gwl

    agreement = calc_agreement(
        {model: difference for model, difference in differences.items() if model != 'Multi-Model Median'},
        as_datatree=as_datatree
    )

    if as_datatree:
        tree_dict = {
            f"/{model}/{expt}/{var}": ds
            for model, expt_dict in differences.items()
            for expt, var_dict in expt_dict.items()
            for var, ds in var_dict.items()
        }
        differences = xr.DataTree.from_dict(tree_dict)
    else:
        differences = utils.defaultdict_to_dict(differences)

    return differences, agreement

