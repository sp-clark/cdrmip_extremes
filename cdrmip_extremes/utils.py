"""
utils.py
"""

import xarray as xr
import numpy as np
import cftime
from collections import defaultdict

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


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def _map_leaves(nested_dict, func):
    out = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            out[k] = _map_leaves(v, func)
        else:
            out[k] = func(v)
    return out

def map_nested(func, *trees):
    """Apply func to the leaves of one or more identically nested dicts.
    Leaves returning None, and branches left empty, are dropped."""
    if isinstance(trees[0], dict):
        out = {k: map_nested(func, *(t[k] for t in trees)) for k in trees[0]}
        return {k: v for k, v in out.items()
                if v is not None and not (isinstance(v, dict) and not v)}
    return func(*trees)
    
def map_with_picontrol(func, tree, pi_tree):
    """Apply func(ds, pi_leaf) over {model: {expt: {member: ds}}}, taking pi_leaf
    from {model: {member: ...}}. Members without a piControl entry are dropped."""
    return {
        model: {
            expt: {mem: func(ds, pi_tree[model][mem])
                   for mem, ds in mems.items() if mem in pi_tree.get(model, {})}
            for expt, mems in expts.items()
        }
        for model, expts in tree.items()
    }

    
_UNIT_FIXES = {
    'kg m-2 s-1': ('mm/day', 86400),
}

def fix_units(ds):
    """Apply known unit conversions (e.g. pr: kg m-2 s-1 -> mm/day) in place."""
    for var in ds.data_vars:
        units = ds[var].attrs.get('units')
        if units in _UNIT_FIXES:
            new_units, factor = _UNIT_FIXES[units]
            ds = ds.assign({var: ds[var] * factor})
            ds[var].attrs['units'] = new_units
    return ds
    
def select_years(ds, lo, hi):
    """Keep years lo..hi inclusive."""
    years = ds.time.dt.year
    return ds.isel(time=(years >= lo) & (years <= hi))
# ---------------------------------------------------------------------------
# mutli-member/multi-model reduction
# ---------------------------------------------------------------------------

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

def reduce_member_tree(nested_dict, reduction='median', drop_coords=None):
    """
    Reduce a {model: {expt: {member: ds}}} tree over its member dimension.
    Returns {model: {expt: ds}}.
    """
    output = defaultdict(dict)
    for model, expt_dt in nested_dict.items():
        for expt, member_dt in expt_dt.items():
            output[model][expt] = reduce_members(
                member_dt, reduction=reduction, drop_coords=drop_coords,
            )
    return dict(output)
 
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
    
def member_median(tree):
    """{model: {expt: {member: da}}} -> {model: {expt: da}}, median across members."""
    return {model: {expt: xr.concat(list(members.values()), dim="member").median("member")
                    for expt, members in expts.items()}
            for model, expts in tree.items()}

def add_model_reduction(model_dt, reduction="median"):
    """{model: da} -> same dict plus a multi_model_{reduction} entry."""
    key = f"multi_model_{reduction}"
    models = [da for m, da in model_dt.items() if m != key]
    combined = xr.concat(models, dim="model")
    return {**model_dt, key: getattr(combined, reduction)(dim="model")}
# ---------------------------------------------------------------------------
# CO2 concentrations
# ---------------------------------------------------------------------------

def build_co2_path(piCo2=284.7):
    CO2_4x = piCo2 * 4
    years_up = np.arange(0,140,1)
    concs_up = piCo2 * np.power(1.01, years_up)
    concs_up = np.append(concs_up, CO2_4x)
    years_up = np.append(years_up, 139.321)
    years_down = np.array(139.321)
    years_down = np.append(years_down,np.arange(140,280, 1))
    concs_down = concs_up[0:-1][::-1]
    concs = np.concatenate((concs_up,concs_down),axis=None)
    years = np.concatenate((years_up,years_down),axis=None)
    PI = piCo2 * np.ones(100)
    yearsPI = np.arange(0,100,1)
    const = piCo2 * np.ones(60)
    years_const = np.arange(281,341,1)
    CO2_series = np.concatenate(
        [PI,
         concs[:140],
         concs[140:],
         const],
        axis=None
    )
    years = np.concatenate(
        [yearsPI,
         years[:140]+100,
         years[141:]+100,
         years_const+100,
        ],
        axis=None
    )
    return xr.DataArray(data=CO2_series,
             dims=['year'],
             coords={'year':years}
            )