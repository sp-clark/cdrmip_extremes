import numpy as np
import xarray as xr
import pandas as pd
import os
import netCDF4 as nc
import cftime
from pathlib import Path
from collections import defaultdict
import sys

def find_CMIP_files(activity_id=None,
                    experiment_id=None,
                    table_id=None,
                    variable_id=None,
                    source_id=None,
                    institution_id=None,
                    member_id=None,
                    grid_label=None):

    """
    Flexible CMIP6 search for NCI archive.

    Each argument may be:
        - None  (no filtering)
        - str   (single value)
        - list  (multiple allowed values)
    """

    # ---------- helper ----------
    def make_list(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    activity_id = make_list(activity_id)
    experiment_id = make_list(experiment_id)
    table_id = make_list(table_id)
    variable_id = make_list(variable_id)
    source_id = make_list(source_id)
    institution_id = make_list(institution_id)
    member_id = make_list(member_id)
    grid_label = make_list(grid_label)

    base_dirs = [
        Path('/g/data/fs38/publications/CMIP6'),
        Path('/g/data/oi10/replicas/CMIP6')
    ]

    results = []

    # ---------- traversal ----------
    for base_dir in base_dirs:

        if not base_dir.exists():
            continue

        activities = activity_id if activity_id else [p.name for p in base_dir.iterdir() if p.is_dir()]

        for act in activities:
            act_path = base_dir / act
            if not act_path.exists():
                continue

            institutions = institution_id if institution_id else [p.name for p in act_path.iterdir() if p.is_dir()]

            for inst in institutions:
                inst_path = act_path / inst
                if not inst_path.exists():
                    continue

                models = source_id if source_id else [p.name for p in inst_path.iterdir() if p.is_dir()]

                for model in models:
                    model_path = inst_path / model
                    if not model_path.exists():
                        continue

                    experiments = experiment_id if experiment_id else [p.name for p in model_path.iterdir() if p.is_dir()]

                    for expt in experiments:
                        expt_path = model_path / expt
                        if not expt_path.exists():
                            continue

                        # ---------- member level ----------
                        members = member_id if member_id else [
                            p.name for p in expt_path.iterdir() if p.is_dir()
                        ]

                        for mem in members:

                            member_path = expt_path / mem
                            if not member_path.exists():
                                continue

                            tables = table_id if table_id else [
                                p.name for p in member_path.iterdir() if p.is_dir()
                            ]

                            for table in tables:
                                table_path = member_path / table
                                if not table_path.exists():
                                    continue

                                variables = variable_id if variable_id else [
                                    p.name for p in table_path.iterdir() if p.is_dir()
                                ]

                                for var in variables:
                                    var_path = table_path / var
                                    if not var_path.exists():
                                        continue

                                    grid_labels = grid_label if grid_label else [
                                        p.name for p in var_path.iterdir() if p.is_dir()
                                    ]
                                    
                                    for grid in grid_labels:
                                    
                                        grid_path = var_path / grid
                                        if not grid_path.exists():
                                            continue
                                    
                                        versions = [
                                            v for v in grid_path.iterdir()
                                            if v.is_dir() and v.name != 'files'
                                        ]
                                    
                                        if not versions:
                                            continue
                                    
                                        latest_version = max(
                                            versions,
                                            key=lambda v: v.stat().st_mtime
                                        )
                                    
                                        netcdf_files = sorted(str(f) for f in latest_version.glob("*.nc"))
                                    
                                        if netcdf_files:
                                            results.append({
                                                'activity_id': act,
                                                'institution_id': inst,
                                                'source_id': model,
                                                'experiment_id': expt,
                                                'member_id': mem,
                                                'table_id': table,
                                                'variable_id': var,
                                                'grid_label': grid,
                                                'version': latest_version.name,
                                                'path': netcdf_files
                                            })

    return pd.DataFrame(results)

def search_and_load(experiment_id, table_id, variable_id, vars_to_drop, 
                                chunks, threshold=None, check_end_year=True):
    print(f"Searching for {experiment_id} files")
    search_df = find_CMIP_files(
        experiment_id=experiment_id,
        table_id=table_id,
        variable_id=variable_id
    ).sort_values(by=['source_id', 'experiment_id', 'member_id'])

    data = defaultdict(dict)
    for row in search_df.itertuples():
        print(f"Loading {row.source_id} {row.member_id}")
        paths = [p for p in row.path if row.version in p]
        ds = xr.open_mfdataset(
            paths, chunks=chunks, drop_variables=vars_to_drop,
            use_cftime=True, combine="nested", concat_dim='time'
        )
        print(f"{row.source_id} {row.member_id}: {ds.time.values[0]} → {ds.time.values[-1]}")

        if check_end_year:
            end_year = ds.time.values[-1].year
            if end_year > threshold:
                data[row.source_id][row.member_id] = ds
            else:
                print(f"  Skipping — end year {end_year} not beyond {threshold}")
        else:
            data[row.source_id][row.member_id] = ds

    return dict(data)

def search_and_load_parent(child_ds,chunks,vars_to_drop):
    experiment_id = child_ds.attrs['experiment_id']
    variant_label = child_ds.attrs['variant_label']
    source_id = child_ds.attrs['source_id']
    print(f"Seaching for {source_id} {experiment_id} {variant_label} parent")
    parent_experiment_id = child_ds.attrs['parent_experiment_id']
    parent_variant_label = child_ds.attrs['parent_variant_label']
    if source_id == 'GISS-E2-2-G':
        parent_source_id = 'GISS-E2-2-G'
    else:
        parent_source_id = child_ds.attrs['parent_source_id']
    grid_label = child_ds.attrs['grid_label']
    table_id = child_ds.attrs['table_id']
    variable_id = child_ds.attrs['variable_id']
    search = find_CMIP_files(
                experiment_id = parent_experiment_id,
                source_id = parent_source_id,
                member_id = parent_variant_label,
                table_id = table_id,
                variable_id = variable_id,
                grid_label=grid_label,
    )
    if len(search) == 0:
        print(f"No parent found for {source_id} {experiment_id} {variant_label} "
              f"(parent: {parent_source_id} {parent_experiment_id} {parent_variant_label}) — skipping")
        return None
    print(f"Loading {parent_source_id} {parent_experiment_id} {parent_variant_label}")
    path = search['path'][0]
    parent_ds = xr.open_mfdataset(
        path, chunks=chunks,drop_variables=vars_to_drop,
        use_cftime=True, combine='nested', concat_dim= 'time'
    )
    return parent_ds

def concatenate_branches(child_ds,parent_ds,time_dim='time',branch_date=None):

    parent_times = parent_ds.indexes[time_dim]
    calendar = parent_times.calendar
    # determine branch date
    if branch_date is None:
        # convert branch_time_in_parent (days) to datetime
        branch_days = child_ds.branch_time_in_parent
        units = child_ds.parent_time_units
        branch_date = cftime.num2date(branch_days, units=units, calendar=calendar)
    
    ds_concat = xr.concat([parent_ds.sel({time_dim:slice(None,branch_date)}),child_ds],dim=time_dim)
    
    return ds_concat
