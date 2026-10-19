import os
import xarray as xr
from pathlib import Path
import json

from .configs import PROC_DIR, CDR_MODELS
from . import utils

def _to_dataset(leaf, name):
    if isinstance(leaf, xr.DataTree):
        return leaf.ds
    if isinstance(leaf, xr.DataArray):
        return leaf.to_dataset(name=leaf.name or name)
    return leaf


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
                _to_dataset(leaf, var).to_netcdf(path)


def load_tree(proc_dir, var, activity_id, category=None, chunks="auto",
              as_datatree=False, as_dataarray=False, model_order=None, expt=None):
    """Reload the tree saved by save_tree, keyed the same way.
    If `as_datatree` is True, each leaf is returned as an xr.DataTree;
    otherwise as an xr.Dataset.
    If `expt` is given (str or list of str), only those experiments are loaded.
    """
    expts = [expt] if isinstance(expt, str) else expt
    load_dir = Path(proc_dir) / f"{activity_id}/{var}/"
    suffix = f"_{var}_{category}.nc" if category is not None else f"_{var}.nc"
    data_dt = {}
    for model_dir in sorted(p for p in load_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        for expt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            e = expt_dir.name
            if expts is not None and e not in expts:
                continue
            prefix = f"{model}_{e}_"
            for path in sorted(expt_dir.glob(f"*{suffix}")):
                member = path.name[len(prefix):-len(suffix)]
                if as_dataarray:
                    leaf = xr.open_dataarray(path, chunks=chunks)
                else:
                    ds = xr.open_dataset(path, chunks=chunks)
                    leaf = xr.DataTree(dataset=ds) if as_datatree else ds
                data_dt.setdefault(model, {}).setdefault(e, {})[member] = leaf
    if model_order is not None:
        data_dt = ({m: data_dt[m] for m in model_order if m in data_dt}
                   | {m: v for m, v in data_dt.items() if m not in model_order})
    return data_dt

def _to_builtin(x):
    return getattr(x, 'values', x).item()

def _deep_update(d, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

def save_peak_warming(peak_warming, activity_id, proc_dir):
    """Save peak_warming, merging into any existing file (new values win)."""
    path = Path(proc_dir) / activity_id / 'peak_warming' / 'peak_warming.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(_deep_update(existing, peak_warming), default=_to_builtin))

def load_peak_warming(activity_id, proc_dir):
    """Load a peak_warming dict saved by save_peak_warming()."""
    with open(Path(proc_dir) / activity_id / 'peak_warming' / "peak_warming.json") as f:
        return json.load(f)

def save_nested(tree, save_dir, name, overwrite=False, _keys=()):
    """Save a nested dict of Datasets/DataArrays/DataTrees of any depth.
    Every key but the last becomes a directory; all keys go in the filename."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            save_nested(v, save_dir, name, overwrite, _keys + (k,))
        return
    path = Path(save_dir, *_keys[:-1]) / f"{'_'.join(_keys)}_{name}.nc"
    if path.exists() and not overwrite:
        print(f"Skipping (already exists): {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    leaf = _to_dataset(tree, 'data')
    print(f"Saving: {path}")
    leaf.to_netcdf(path)


def _reorder(d, order):
    """Put keys in `order` first (skipping any not present), the rest after."""
    if order is None:
        return d
    return ({k: d[k] for k in order if k in d}
            | {k: v for k, v in d.items() if k not in order})


def load_nested(load_dir, name, chunks='auto', order=None, _keys=()):
    """Reload a tree written by save_nested, keyed the same way.
    `order` : list of str, optional
        Order for the top-level keys (e.g. models); unlisted keys follow
        alphabetically.
    """
    load_dir = Path(load_dir)
    out = {d.name: load_nested(d, name, chunks, _keys=_keys + (d.name,))
           for d in sorted(p for p in load_dir.iterdir() if p.is_dir())}
    prefix = '_'.join(_keys) + '_' if _keys else ''
    suffix = f"_{name}.nc"
    for path in sorted(load_dir.glob(f"{prefix}*{suffix}")):
        out[path.name[len(prefix):-len(suffix)]] = xr.open_dataset(path, chunks=chunks)
    return _reorder(out, order)


DEFAULT_LOADERS = {
    'gsat':               lambda load, mip: load('gsat'),
    'gsat_rolling':       lambda load, mip: load('gsat_rolling'),
    'crossing_years':     lambda load, mip: load('crossing_years', as_dataarray=True),
    'gwl_matching_years': lambda load, mip: load('gwl_matching_years', as_dataarray=True, drop_coords=['gwl']),
    'tas':                lambda load, mip: load('amoc', category='anomaly'),
    'peak_warming':       lambda load, mip: load_peak_warming(mip, PROC_DIR),
}


def load_mip(mip, models, skip=(), reduce_members=True, loaders=DEFAULT_LOADERS):
    def load(var, drop_coords=None, **kw):
        tree = load_tree(PROC_DIR, var, mip, model_order=models, **kw)
        if not reduce_members:
            return tree
        return utils.reduce_member_tree(tree, drop_coords=drop_coords) if drop_coords else utils.reduce_member_tree(tree)

    return {k: f(load, mip) for k, f in loaders.items() if k not in skip}
