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

def save_pattern_tree(pattern_dicts, proc_dir, subdir='patterns', vars=None):
    """Save {pattern_name: {model: {expt: {member: {var: da}}}}} to disk.

    vars: None saves all variables; a string or iterable of strings saves
    only those variables (e.g. vars='slopes' or vars=['slopes', 'intercepts']).
    """
    save_dir = Path(proc_dir) / subdir
    if isinstance(vars, str):
        vars = {vars}
    elif vars is not None:
        vars = set(vars)
    for pattern_name, pattern_dict in pattern_dicts.items():
        for model, expt_dict in pattern_dict.items():
            model_dir = save_dir / pattern_name / model
            
            for expt, member_dict in expt_dict.items():
                for member, var_dict in member_dict.items():
                    for var, da in var_dict.items():
                        if vars is not None and var not in vars:
                            continue
                        model_dir.mkdir(parents=True, exist_ok=True)
                        path = model_dir / f"{model}_{expt}_{member}_{pattern_name}_{var}.nc"
                        if path.exists():
                            print(f"Skipping (already exists): {path}")
                            continue
                        print(f"Saving: {path}")
                        da.to_netcdf(path)

def load_pattern_tree(proc_dir, pattern_names, subdir='patterns', chunks='auto'):
    """Reload the tree saved by save_pattern_tree, keyed the same way."""
    load_dir = Path(proc_dir) / subdir
    pattern_dicts = {}
    for pattern_name in pattern_names:
        pattern_dict = {}
        for model_dir in sorted(p for p in (load_dir / pattern_name).iterdir() if p.is_dir()):
            model = model_dir.name
            prefix = f"{model}_"
            for path in sorted(model_dir.glob(f"{model}_*_{pattern_name}_*.nc")):
                stem = path.stem[len(prefix):]              # "{expt}_{member}_{pattern_name}_{var}"
                front, var = stem.rsplit(f"_{pattern_name}_", 1)
                expt, member = front.split('_', 1)           # assumes expt has no underscore
                da = xr.open_dataarray(path, chunks=chunks)
                pattern_dict.setdefault(model, {}).setdefault(expt, {}).setdefault(member, {})[var] = da
        pattern_dicts[pattern_name] = pattern_dict
    return pattern_dicts

def save_error_tree(error_dicts, proc_dir, subdir='pattern_errors', overwrite=False):
    """Save pattern-error trees to disk.
 
    `error_dicts` maps a kind label to a tree, e.g.
        {'errors': errors_by_pattern, 'rmse': rmse_by_pattern}
    where each tree is nested
        {pattern_type: {predictor_expt: {model: {target_expt: {member: ds}}}}}
    and each leaf is an xr.Dataset (e.g. holding `tas`).
 
    Layout on disk (every structural key is a directory, so hyphens/underscores
    in model names or experiment ids can't be mis-parsed on reload):
        {subdir}/{kind}/{pattern_type}/{predictor_expt}/{model}/{target_expt}/
            {model}_{predictor_expt}_{target_expt}_{member}_{kind}.nc
    """
    save_dir = Path(proc_dir) / subdir
    for kind, tree in error_dicts.items():
        for pattern_type, pred_dict in tree.items():
            for predictor_expt, model_dict in pred_dict.items():
                for model, target_dict in model_dict.items():
                    for target_expt, member_dict in target_dict.items():
                        out_dir = (save_dir / kind / pattern_type /
                                   predictor_expt / model / target_expt)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        for member, ds in member_dict.items():
                            fname = (f"{model}_{predictor_expt}_{target_expt}_"
                                     f"{member}_{kind}.nc")
                            path = out_dir / fname
                            if path.exists() and not overwrite:
                                print(f"Skipping (already exists): {path}")
                                continue
                            print(f"Saving: {path}")
                            _to_dataset(ds).to_netcdf(path)
 
 
def load_error_tree(proc_dir, kinds, subdir='pattern_errors', chunks='auto',
                    pattern_types=None):
    """Reload the trees saved by save_error_tree, keyed the same way.
 
    Returns {kind: {pattern_type: {predictor_expt: {model: {target_expt:
             {member: ds}}}}}} for each requested kind (e.g. ['errors','rmse']).
 
    pattern_types: None loads all pattern types; a string or iterable of
    strings loads only those (e.g. pattern_types='global' or
    pattern_types=['global', 'regional']).
    """
    load_dir = Path(proc_dir) / subdir
    if isinstance(pattern_types, str):
        pattern_types = {pattern_types}
    elif pattern_types is not None:
        pattern_types = set(pattern_types)
    out = {}
    for kind in kinds:
        tree = {}
        kind_dir = load_dir / kind
        for pt_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            if pattern_types is not None and pt_dir.name not in pattern_types:
                continue
            for pred_dir in sorted(p for p in pt_dir.iterdir() if p.is_dir()):
                for model_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
                    model = model_dir.name
                    for tgt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                        prefix = f"{model}_{pred_dir.name}_{tgt_dir.name}_"
                        suffix = f"_{kind}.nc"
                        for path in sorted(tgt_dir.glob(f"*{suffix}")):
                            member = path.name[len(prefix):-len(suffix)]
                            ds = xr.open_dataset(path, chunks=chunks)
                            (tree.setdefault(pt_dir.name, {})
                                 .setdefault(pred_dir.name, {})
                                 .setdefault(model, {})
                                 .setdefault(tgt_dir.name, {})[member]) = ds
        out[kind] = tree
    return out