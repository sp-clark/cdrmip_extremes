"""
CMIP6 file search
"""

from collections import defaultdict
from pathlib import Path

import cftime
import pandas as pd
import xarray as xr

from .configs import RAW_DIR

# Set these to the location of the relevant CMIP6 data
DEFAULT_BASE_DIRS = [
    # RAW_DIR,
    Path("/g/data/fs38/publications/CMIP6"),
    Path("/g/data/oi10/replicas/CMIP6"),
    Path("/g/data/if69/sc1326/esgf_cache/CMIP6"),
]


def find_CMIP_files(activity_id=None,
                     experiment_id=None,
                     table_id=None,
                     variable_id=None,
                     source_id=None,
                     institution_id=None,
                     member_id=None,
                     grid_label=None,
                     base_dirs=None):
    """
    Flexible CMIP6 search

    Each argument may be:
        - None  (no filtering)
        - str   (single value)
        - list  (multiple allowed values)

    Returns
    -------
    pandas.DataFrame
        One row per (activity, institution, source, experiment, member,
        table, variable, grid) combination found, with a 'path' column
        listing the matching NetCDF files for the latest version.
    """

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

    base_dirs = base_dirs or DEFAULT_BASE_DIRS
    results = []

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

                        members = member_id if member_id else [p.name for p in expt_path.iterdir() if p.is_dir()]

                        for mem in members:
                            member_path = expt_path / mem
                            if not member_path.exists():
                                continue

                            tables = table_id if table_id else [p.name for p in member_path.iterdir() if p.is_dir()]

                            for table in tables:
                                table_path = member_path / table
                                if not table_path.exists():
                                    continue

                                variables = variable_id if variable_id else [p.name for p in table_path.iterdir() if p.is_dir()]

                                for var in variables:
                                    var_path = table_path / var
                                    if not var_path.exists():
                                        continue

                                    grid_labels = grid_label if grid_label else [p.name for p in var_path.iterdir() if p.is_dir()]

                                    for grid in grid_labels:
                                        grid_path = var_path / grid
                                        if not grid_path.exists():
                                            continue

                                        versions = [
                                            v for v in grid_path.iterdir()
                                            if v.is_dir() and v.name != "files"
                                        ]
                                        if not versions:
                                            continue

                                        latest_version = max(versions, key=lambda v: v.stat().st_mtime)
                                        netcdf_files = sorted(str(f) for f in latest_version.glob("*.nc"))

                                        if netcdf_files:
                                            results.append({
                                                "activity_id": act,
                                                "institution_id": inst,
                                                "source_id": model,
                                                "experiment_id": expt,
                                                "member_id": mem,
                                                "table_id": table,
                                                "variable_id": var,
                                                "grid_label": grid,
                                                "version": latest_version.name,
                                                "path": netcdf_files,
                                            })

    return pd.DataFrame(results)    