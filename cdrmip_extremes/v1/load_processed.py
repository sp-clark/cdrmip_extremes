import numpy as np
import xarray as xr
import os
import netCDF4 as nc
import cftime
from pathlib import Path
from collections import defaultdict
import sys

from cdrmip_extremes.configs import data_dir, proc_dir

def load_processed(proc_dir, activity_id, category, varname,
                   chunks=None, as_datatree=True, average_members=False):
    """
    Load processed netCDF files saved by save_processed.
    Parameters
    ----------
    proc_dir         : Path, base processed directory
    activity_id      : str, e.g. 'ScenarioMIP', 'flat10MIP'
    category         : str, e.g. 'anomalies', 'regridded'
    varname          : str, e.g. 'tas', 'pr'
    chunks           : dict, optional dask chunks e.g. {'time': 120}
    as_datatree      : bool, return xr.DataTree if True, else nested dict
    average_members  : bool, if True and multiple members exist for a
                       model/experiment, average across them before returning
    Returns
    -------
    xr.DataTree or nested dict { model: { expt: { member: ds } } }
    If average_members=True, member level is collapsed:
    xr.DataTree or nested dict { model: { expt: ds } }
    """
    save_dir = proc_dir / activity_id / varname / category
    if not save_dir.exists():
        raise FileNotFoundError(f"No processed data found at {save_dir}")
    data = {}
    for model_dir in sorted(save_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for path in sorted(model_dir.glob(f"*_{varname}_{category}.nc")):
            stem = path.stem
            suffix = f"_{varname}_{category}"
            prefix = f"{model}_"
            remainder = stem[len(prefix):-len(suffix)]  # e.g. ssp126_r1i1p1f1
            if '_' in remainder:
                expt, member = remainder.rsplit('_', 1)
            else:
                expt = remainder
                member = None
            print(f"Loading: {path}")
            ds = xr.open_dataset(path, chunks=chunks, use_cftime=True)
            if member is not None:
                data.setdefault(model, {}).setdefault(expt, {})[member] = ds
            else:
                data.setdefault(model, {})[expt] = ds

    if average_members:
        for model, expt_dict in data.items():
            for expt, member_or_ds in expt_dict.items():
                if isinstance(member_or_ds, dict):
                    datasets = list(member_or_ds.values())
                    if len(datasets) > 1:
                        expt_dict[expt] = xr.concat(datasets, dim='member').mean(dim='member')
                    else:
                        expt_dict[expt] = datasets[0]

    if as_datatree:
        tree_dict = {}
        for model, expt_dict in data.items():
            for expt, member_or_ds in expt_dict.items():
                if isinstance(member_or_ds, dict):
                    for member, ds in member_or_ds.items():
                        tree_dict[f"/{model}/{expt}/{member}"] = ds
                else:
                    tree_dict[f"/{model}/{expt}"] = member_or_ds
        return xr.DataTree.from_dict(tree_dict)
    else:
        return data


def load_ext_data(activity_id, varnames=None,
                      chunks=None, as_datatree=True):
    """
    Load exceedance netCDF files saved in the extremes structure:
    proc_dir / activity_id / "extremes" / varname / model / expt / f"{model}_{expt}_{varname}.nc"

    Parameters
    ----------
    proc_dir    : Path, base processed directory
    activity_id : str, e.g. 'ScenarioMIP'
    varnames    : str, list of str, or None
                  If None, loads all variable folders found under "extremes".
                  If a string, loads just that one variable.
                  If a list, loads each of those variables.
    chunks      : dict, optional dask chunks e.g. {'time': 120}
    as_datatree : bool, return xr.DataTree if True, else nested dict

    Returns
    -------
    xr.DataTree or nested dict { model: { expt: { varname: ds } } }
    """
    RENAME_MAP = {
    "median": "Multi-Model Median",
    }
    
    
    extremes_dir = proc_dir / activity_id / "extremes"
    if not extremes_dir.exists():
        raise FileNotFoundError(f"No processed data found at {extremes_dir}")

    if varnames is None:
        var_dirs = sorted(d for d in extremes_dir.iterdir() if d.is_dir())
    elif isinstance(varnames, str):
        var_dirs = [extremes_dir / varnames]
    else:
        var_dirs = [extremes_dir / v for v in varnames]

    data = {}
    for var_dir in var_dirs:
        varname = var_dir.name
        if not var_dir.exists():
            print(f"Skipping missing variable dir: {var_dir}")
            continue

        for model_dir in sorted(var_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            for expt_dir in sorted(model_dir.iterdir()):
                if not expt_dir.is_dir():
                    continue
                expt = expt_dir.name
                path = expt_dir / f"{model}_{expt}_{varname}.nc"
                if not path.exists():
                    continue
                print(f"Loading: {path}")
                ds = xr.open_dataset(path, chunks=chunks, use_cftime=True)
                model_name = RENAME_MAP.get(model, model)
                data.setdefault(model_name, {}).setdefault(expt, {})[varname] = ds

    if as_datatree:
        tree_dict = {
            f"/{model}/{expt}/{varname}": ds
            for model, expt_dict in data.items()
            for expt, var_dict in expt_dict.items()
            for varname, ds in var_dict.items()
        }
        return xr.DataTree.from_dict(tree_dict)
    else:
        return data

def load_gwl_differences(proc_dir, activity_id, category, varname,
                          chunks=None, as_datatree=True):
    """
    Load GWL-difference files saved with layout:
    proc_dir / activity_id / category / model / expt / {model}_{expt}_{varname}_gwl_difference.nc
    """
    save_dir = proc_dir / activity_id / category
    if not save_dir.exists():
        raise FileNotFoundError(f"No processed data found at {save_dir}")

    data = {}
    for model_dir in sorted(save_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for expt_dir in sorted(model_dir.iterdir()):
            if not expt_dir.is_dir():
                continue
            expt = expt_dir.name
            matches = sorted(expt_dir.glob(f"{model}_{expt}_{varname}_gwl_difference.nc"))
            if not matches:
                continue
            path = matches[0]
            print(f"Loading: {path}")
            ds = xr.open_dataset(path, chunks=chunks, use_cftime=True)
            data.setdefault(model, {})[expt] = ds

    if as_datatree:
        tree_dict = {
            f"/{model}/{expt}": ds
            for model, expt_dict in data.items()
            for expt, ds in expt_dict.items()
        }
        return xr.DataTree.from_dict(tree_dict)
    else:
        return data
