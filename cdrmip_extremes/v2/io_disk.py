import os
import xarray as xr
from pathlib import Path
import json

from .configs import PROC_DIR

def _to_dataset(obj):
    """Return an xr.Dataset from either a DataTree leaf or a Dataset."""
    return obj.ds if hasattr(obj, "ds") else obj


def save_tree(data_dt, proc_dir, var, activity_id, category=None, overwrite=False):
    """Save a nested {model: {expt: {member: leaf}}} structure to disk.
    `leaf` may be an xr.DataTree (its .ds is written) or an xr.Dataset.
    If `overwrite` is False, existing files are skipped; if True, they're re-written.
    """
    save_dir = Path(proc_dir) / f"{activity_id}/{var}/"
    for model, expt_dt in data_dt.items():
        model_dir = save_dir / model
        for expt, member_dt in expt_dt.items():
            expt_dir = model_dir / expt
            os.makedirs(expt_dir, exist_ok=True)
            for member, leaf in member_dt.items():
                if category is not None:
                    path = expt_dir / f"{model}_{expt}_{member}_{var}_{category}.nc"
                else:
                    path = expt_dir / f"{model}_{expt}_{member}_{var}.nc"
                if path.exists() and not overwrite:
                    print(f"Skipping (already exists): {path}")
                    continue
                print(f"Saving: {path}")
                _to_dataset(leaf).to_netcdf(path)


def load_tree(proc_dir, var, activity_id, category=None, chunks="auto",
              as_datatree=False):
    """Reload the tree saved by save_tree, keyed the same way.
    If `as_datatree` is True, each leaf is returned as an xr.DataTree;
    otherwise as an xr.Dataset.
    """
    load_dir = Path(proc_dir) / f"{activity_id}/{var}/"
    data_dt = {}
    for model_dir in sorted(p for p in load_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        for expt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            expt = expt_dir.name
            prefix = f"{model}_{expt}_"
            suffix = f"_{var}_{category}.nc" if category is not None else f"_{var}.nc"
            for path in sorted(expt_dir.glob(f"*{suffix}")):
                member = path.name[len(prefix):-len(suffix)]
                ds = xr.open_dataset(path, chunks=chunks)
                leaf = xr.DataTree(dataset=ds) if as_datatree else ds
                data_dt.setdefault(model, {}).setdefault(expt, {})[member] = leaf
    return data_dt

def load_peak_warming(out_dir=None):
    """Load a peak_warming dict saved by save_peak_warming()."""
    out_dir = Path(out_dir) if out_dir is not None else PROC_DIR / 'ScenarioMIP/peak_warming'
    store = out_dir / "peak_warming.json"
    with open(store) as f:
        return json.load(f)

