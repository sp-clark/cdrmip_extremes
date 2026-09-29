"""
save.py
"""

import os
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _save_nc(ds, path, overwrite=False):
    """
    """
    path = Path(path)
    if path.exists() and not overwrite:
        print(f"Skipping (already exists): {path}")
        return

    print(f"Saving: {path}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp_path)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_processed(data, proc_dir, activity_id, varname, category=None, overwrite=False):
    """
    Save a nested DataTree or dict (model/expt or model/expt/member) of
    Dataset/DataArray leaves to netCDF, under:
        proc_dir/activity_id/varname/category/{model}/{model}_{expt}[_{member}]_{varname}_{category}.nc

    Parameters
    ----------
    data     : xr.DataTree or nested dict
    proc_dir : Path, base processed directory
    activity_id : str, e.g. 'CDRMIP', 'ScenarioMIP'
    category : str, e.g. 'concat', 'anomaly', 'gsat'
    varname  : str, name to use if a leaf is a bare DataArray (e.g. 'tas', 'gsat')
    overwrite : bool
        If False (default), skip any file that already exists -- lets you
        resume an interrupted save. Set True after any upstream config
        change, since otherwise stale files from before the change won't
        be regenerated.
    """
    if category is not None:
        save_dir = Path(proc_dir) / activity_id / category
    else:
        save_dir = Path(proc_dir) / activity_id
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, xr.DataTree):
        get_children = lambda node: node.children.items()
        get_ds = lambda node: node.ds
        is_leaf = lambda node: node.is_leaf
    else:
        get_children = lambda node: node.items()
        get_ds = lambda node: node.to_dataset(name=varname) if isinstance(node, xr.DataArray) else node
        is_leaf = lambda node: isinstance(node, (xr.Dataset, xr.DataArray))

    items = data.children.items() if isinstance(data, xr.DataTree) else data.items()
    for model, expt_node in items:
        model_dir = save_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for expt, node in get_children(expt_node):
            if is_leaf(node):
                path = model_dir / f"{model}_{expt}_{varname}.nc"
                _save_nc(get_ds(node), path, overwrite=overwrite)
            else:
                for member, leaf in get_children(node):
                    if not is_leaf(leaf):
                        print(f"  Skipping {model}/{expt}/{member}: leaf is {type(leaf)}, "
                              f"not Dataset/DataArray -- use save_scalar_outputs for this category.")
                        continue
                    path = model_dir / f"{model}_{expt}_{member}_{varname}.nc"
                    _save_nc(get_ds(leaf), path, overwrite=overwrite)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_scalar_outputs(data, proc_dir, activity_id, category):
    """
    Save a nested dict of plain scalar dicts
    """
    save_dir = Path(proc_dir) / activity_id / category
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{category}.json"

    if path.exists():
        print(f"Skipping (already exists): {path}")
        return

    print(f"Saving: {path}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    tmp_path.rename(path)