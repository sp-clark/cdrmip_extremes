import xarray as xr
import numpy as np
import os
import sys
import cftime
import xesmf as xe

from cdrmip_extremes.configs import data_dir, proc_dir

# 180 lon, 90 lat grid
lon_new = np.linspace(-179.5, 179.5, 180)
lat_new = np.linspace(-89.5, 89.5, 90)

grid_out = xr.Dataset(
    {
        "lat": (["lat"], lat_new),
        "lon": (["lon"], lon_new),
    }
)

_regridder_cache = {}

def regrid_to_common(ds, method='bilinear', periodic=True):
    # Build a cache key from the source grid shape and bounds
    cache_key = (ds.lat.shape, ds.lon.shape, 
                 float(ds.lat.min()), float(ds.lat.max()),
                 float(ds.lon.min()), float(ds.lon.max()),
                 method)
    
    if cache_key not in _regridder_cache:
        print(f"Building regridder for grid {ds.lat.shape} x {ds.lon.shape}")
        _regridder_cache[cache_key] = xe.Regridder(
            ds, grid_out, method, periodic=periodic, reuse_weights=False
        )
    
    return _regridder_cache[cache_key](ds)

def save_processed(data, proc_dir, activity_id, category, varname):
    """
    Save a nested DataTree or dict (model/expt or model/expt/member) to netCDF.
    
    Parameters
    ----------
    data     : xr.DataTree or nested dict
    proc_dir : Path, base processed directory
    category : str, e.g. 'anomalies', 'regridded'
    varname  : str, e.g. 'tas', 'pr'
    """
    save_dir = proc_dir / activity_id / varname / category
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data, xr.DataTree):
        get_children = lambda node: node.children.items()
        get_ds = lambda node: node.ds
        is_leaf = lambda node: node.is_leaf
    else:
        get_children = lambda node: node.items()
        get_ds = lambda node: node.to_dataset(name=varname) if isinstance(node, xr.DataArray) else node
        is_leaf = lambda node: isinstance(node, (xr.Dataset, xr.DataArray))

    for model, expt_node in (data.children.items() if isinstance(data, xr.DataTree) else data.items()):
        model_dir = save_dir / model
        os.makedirs(model_dir, exist_ok=True)
        for expt, node in get_children(expt_node):
            if is_leaf(node):
                # two-level tree: save directly
                path = model_dir / f"{model}_{expt}_{varname}_{category}.nc"
                _save_nc(get_ds(node), path)
            else:
                # three-level tree: iterate members
                for member, leaf in get_children(node):
                    path = model_dir / f"{model}_{expt}_{member}_{varname}_{category}.nc"
                    _save_nc(get_ds(leaf), path)

def _save_nc(ds, path):
    if path.exists():
        print(f"Skipping (already exists): {path}")
        return
    print(f"Saving: {path}")
    ds.to_netcdf(path)