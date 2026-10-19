from collections import defaultdict
from pathlib import Path

import cftime
import pandas as pd
import xarray as xr

from .file_search import find_CMIP_files
from .processing import regrid_to_common
from .configs import DEFAULT_CHUNKS, DEFAULT_VARS_TO_DROP

# ---------------------------------------------------------------------------
# Search and Load Specific CMIP6 datasets
# ---------------------------------------------------------------------------

def search_and_load(nest_by=('source_id', 'member_id'),
                     vars_to_drop=None,
                     chunks=None,
                     threshold=None,
                     check_end_year=True,
                     warn_on_overwrite=True,
                     **search_kwargs):
    """
    Search the CMIP6 archive and load matching files into a nested dict.
 
    Parameters
    ----------
    nest_by : tuple/list of str
        Column names (from find_CMIP_files' output) defining the nesting
        order of the returned dict, from outermost to innermost key.
        e.g. ('source_id',)                      -> {model: ds}
             ('source_id', 'member_id')           -> {model: {member: ds}}
             ('source_id', 'experiment_id', 'member_id')
                                                   -> {model: {expt: {member: ds}}}
        Any column present in the find_CMIP_files results can be used,
        e.g. 'grid_label', 'table_id', 'variable_id', etc.
    vars_to_drop : list or None
        Passed to xr.open_mfdataset as drop_variables.
    chunks : dict or None
        Passed to xr.open_mfdataset.
    threshold : int or None
        Year threshold used when check_end_year=True.
    check_end_year : bool
        If True, skip datasets whose final timestep year is <= threshold.
    warn_on_overwrite : bool
        If True, print a warning when a nesting key collision causes one
        dataset to silently overwrite another (this happens if nest_by
        doesn't uniquely identify each row).
    **search_kwargs
        Forwarded directly to find_CMIP_files (activity_id, experiment_id,
        table_id, variable_id, source_id, institution_id, member_id,
        grid_label).
 
    Returns
    -------
    dict
        Nested dictionary of xarray Datasets, nested according to nest_by.
    """

    chunks = DEFAULT_CHUNKS if chunks is None else chunks
    vars_to_drop = DEFAULT_VARS_TO_DROP if vars_to_drop is None else vars_to_drop
    print(f"Searching for files matching: {search_kwargs}")
    search_df = find_CMIP_files(**search_kwargs)
 
    if search_df.empty:
        print("No matching files found.")
        return {}
 
    missing_cols = [c for c in nest_by if c not in search_df.columns]
    if missing_cols:
        raise ValueError(
            f"nest_by contains columns not produced by find_CMIP_files: {missing_cols}"
        )
 
    search_df = search_df.sort_values(by=list(nest_by))
 
    data = {}
    for row in search_df.itertuples():
        key_values = [getattr(row, key) for key in nest_by]
        label = " / ".join(str(k) for k in key_values)
        print(f"Loading {label}")
 
        paths = [p for p in row.path if row.version in p]
        ds = xr.open_mfdataset(
            paths, chunks=chunks, drop_variables=vars_to_drop,
            use_cftime=True, combine="nested", concat_dim='time'
        )
        if 'time' in ds.dims and 'time' not in ds.coords:
            ds = ds.squeeze('time', drop=True)
        if 'time' in ds.coords:
            print(f"  {ds.time.values[0]} -> {ds.time.values[-1]}")
 
        if check_end_year:
            end_year = ds.time.values[-1].year
            if end_year <= threshold:
                print(f"  Skipping — end year {end_year} not beyond {threshold}")
                continue
 
        # walk/create the nested dict according to nest_by
        d = data
        for key in key_values[:-1]:
            d = d.setdefault(key, {})
 
        last_key = key_values[-1]
        if warn_on_overwrite and last_key in d:
            print(
                f"  WARNING: key '{last_key}' under {label} already exists "
                f"and will be overwritten — nest_by may not uniquely identify this dataset."
            )
        d[last_key] = ds
 
    return data

# ---------------------------------------------------------------------------
# Search and Load Parent Datasets
# ---------------------------------------------------------------------------


def search_and_load_parent(child_ds, chunks=None, vars_to_drop=None,
                            source_id_overrides=None, cache=None):
    """
    Find and load the parent dataset of child_ds, based on its CMIP6
    provenance attributes (parent_experiment_id, parent_variant_label, etc).

    Parameters
    ----------
    source_id_overrides : dict, optional
        {source_id: parent_source_id} for models whose 'parent_source_id'
        attribute is missing/unreliable. Defaults to the GISS-E2-2-G case
    cache : dict, optional
        Pass the same dict across repeated calls (e.g. from the Pipeline)
        to avoid re-searching/re-opening an identical parent multiple times.
    """
    chunks = DEFAULT_CHUNKS if chunks is None else chunks
    vars_to_drop = DEFAULT_VARS_TO_DROP if vars_to_drop is None else vars_to_drop
    
    source_id_overrides = source_id_overrides or {'GISS-E2-2-G': 'GISS-E2-2-G'}

    required_attrs = ['experiment_id', 'variant_label', 'source_id',
                       'parent_experiment_id', 'parent_variant_label',
                       'grid_label', 'table_id', 'variable_id']
    missing = [a for a in required_attrs if a not in child_ds.attrs]
    if missing:
        raise KeyError(f"child_ds is missing expected attrs: {missing}")

    experiment_id = child_ds.attrs['experiment_id']
    variant_label = child_ds.attrs['variant_label']
    source_id = child_ds.attrs['source_id']
    parent_experiment_id = child_ds.attrs['parent_experiment_id']
    parent_variant_label = child_ds.attrs['parent_variant_label']
    grid_label = child_ds.attrs['grid_label']
    table_id = child_ds.attrs['table_id']
    variable_id = child_ds.attrs['variable_id']

    parent_source_id = source_id_overrides.get(
        source_id, child_ds.attrs.get('parent_source_id', source_id)
    )

    print(f"Searching for {source_id} {experiment_id} {variant_label} parent")

    cache_key = (parent_source_id, parent_experiment_id, parent_variant_label,
                 table_id, variable_id, grid_label)
    if cache is not None and cache_key in cache:
        print(f"  Using cached {parent_source_id} {parent_experiment_id} {parent_variant_label}")
        return cache[cache_key]

    search = find_CMIP_files(
        experiment_id=parent_experiment_id,
        source_id=parent_source_id,
        member_id=parent_variant_label,
        table_id=table_id,
        variable_id=variable_id,
        grid_label=grid_label,
    )

    if len(search) == 0:
        print(f"No parent found for {source_id} {experiment_id} {variant_label} "
              f"(parent: {parent_source_id} {parent_experiment_id} {parent_variant_label}) — skipping")
        if cache is not None:
            cache[cache_key] = None
        return None

    if len(search) > 1:
        print(f"  WARNING: {len(search)} matches found for parent "
              f"{parent_source_id} {parent_experiment_id} {parent_variant_label} "
              f"({table_id}/{variable_id}/{grid_label}) — using the first "
              f"(fs38 is checked before oi10)")

    print(f"Loading {parent_source_id} {parent_experiment_id} {parent_variant_label}")
    path = search['path'].iloc[0]
    parent_ds = xr.open_mfdataset(
        path, chunks=chunks, drop_variables=vars_to_drop,
        use_cftime=True, combine='nested', concat_dim='time'
    )

    if cache is not None:
        cache[cache_key] = parent_ds

    return parent_ds

def load_fx(variables,table_id='fx',domain='atm',experiment_id='1pctCO2-cdr',
            source_id='ACCESS-ESM1-5',member_id='r1i1p1f1',grid_label='gn'):
    data = {}
    for var in variables:
        ds = search_and_load(
            experiment_id=experiment_id,
            source_id=source_id,
            member_id=member_id,
            table_id=table_id,
            variable_id=[var],
            grid_label=grid_label,
            vars_to_drop=['time'],
            check_end_year=False,
        )[source_id][member_id]
        if domain == 'atm':
            regridded_ds = regrid_to_common(ds)
            data[var] = regridded_ds
        else:
            data[var] = ds
            
    return data

def load_land_ocean_masks(sftlf=None):
    if sftlf == None:
        sftlf = load_fx(['sftlf'])['sftlf'].sftlf
    land_mask = xr.where(sftlf > 50, 1, 0)   # 1 = land, 0 = ocean
    ocean_mask = xr.where(sftlf <= 50, 1, 0) # 1 = ocean, 0 = land
    return land_mask,ocean_mask