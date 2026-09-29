import re
from pathlib import Path
import xarray as xr
import json

from . import utils
from .load_raw import load_fx
from .configs import PROC_DIR, RAW_DIR, CDR_MODELS

_MEMBER_SUFFIX = re.compile(r'_(?P<member>r\d+i\d+p\d+f\d+)$')

def load_scalar_outputs(proc_dir, activity_id, category):
    path = Path(proc_dir) / activity_id / category / f"{category}.json"
    with open(path) as f:
        return json.load(f)

def load_processed(proc_dir, activity_id, varname,category=None,
                   chunks=None, as_datatree=False, average_members=False,
                  rename_var=None, model_order=None,print_progress=False):
    save_dir = Path(proc_dir) / activity_id / category if category else Path(proc_dir) / activity_id / varname
    if not save_dir.exists():
        raise FileNotFoundError(f"No processed data found at {save_dir}")

    data = {}
    for model_dir in sorted(save_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for path in sorted(model_dir.glob(f"*_{varname}.nc")):
            if print_progress:
                print(f"Loading: {path}")
            stem = path.stem  # e.g. CESM2_1pctCO2-cdr_r1i1p1f1_gwl_matching_years

            # strip known varname suffix
            suffix = f'_{varname}'
            if not stem.endswith(suffix):
                print(f"  Skipping (unexpected stem format): {path.name}")
                continue
            middle = stem[len(f'{model}_'):-len(suffix)]
            # middle is now e.g. '1pctCO2-cdr_r1i1p1f1' or '1pctCO2-cdr'

            # unambiguously extract member via CMIP6 pattern
            m = _MEMBER_SUFFIX.search(middle)
            if m:
                member = m.group('member')
                expt = middle[:m.start()]
            else:
                member = None
                expt = middle

            ds = xr.open_dataset(path, chunks=chunks, use_cftime=True)
            if rename_var:
                # only rename variables that actually exist in this ds --
                # avoids KeyError if different files in the same category
                # happen to have different variable names
                rename_var_safe = {k: v for k, v in rename_var.items() if k in ds}
                if rename_var_safe:
                    ds = ds.rename(rename_var_safe)
            data.setdefault(model, {}).setdefault(expt, {})

            # fix assignment conflict: always keep expt level as a dict;
            # member=None means there was no per-member distinction when saving
            if member is None:
                data[model][expt]['_no_member'] = ds
            else:
                data[model][expt][member] = ds

    # unwrap _no_member placeholder so callers see ds directly, not a dict
    for model, expt_dict in data.items():
        for expt, member_dict in expt_dict.items():
            if set(member_dict.keys()) == {'_no_member'}:
                data[model][expt] = member_dict['_no_member']

    # --- optionally collapse members ---
    if average_members:
        for model, expt_dict in data.items():
            for expt, val in expt_dict.items():
                if isinstance(val, dict):
                    dsets = list(val.values())
                    if len(dsets) > 1:
                        expt_dict[expt] = xr.concat(dsets, dim='member').mean('member')
                    else:
                        expt_dict[expt] = dsets[0]

    # --- convert to DataTree ---
    if as_datatree:
        tree_dict = {}
        for model, expt_dict in data.items():
            for expt, val in expt_dict.items():
                if isinstance(val, dict):
                    for member, ds in val.items():
                        tree_dict[f"/{model}/{expt}/{member}"] = ds
                else:
                    tree_dict[f"/{model}/{expt}"] = val
        return xr.DataTree.from_dict(tree_dict)
    
    if model_order is not None and not as_datatree:
        data = utils.sort_models(data, model_order)

    return data

DEFAULT_OUTPUTS = [
    'thresholds',
    'gwl_period_extremes', 'gwl_period_extremes_matching',
    'exceedances_median_multimodel', 'exceedances_matching_median_multimodel',
    'branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel',
    'agreement', 'agreement_matching',
]

def load_extremes(proc_dir, activity_id, category='extremes',
                  chunks=None, outputs=None, model_order=None):
    """
    Load outputs saved by ExtremesPipeline.save()

    category : str, default 'extremes'
        Top-level subfolder under proc_dir/activity_id/ -- must match
        whatever `category` was passed to ExtremesPipeline.save().
    outputs : list of str, optional
        Which categories to load. Defaults to all nine in DEFAULT_OUTPUTS.
    model_order : list of str, optional
        Applied via utils.sort_models to every model-keyed output.

    Returns
    -------
    dict {output_name: nested_dict}
    """
    base = Path(proc_dir) / activity_id / category
    outputs = outputs or DEFAULT_OUTPUTS
    result = {}

    def _open(path):
        return xr.open_dataset(path, chunks=chunks, use_cftime=True)

    def _ext_type(stem):
        """Last underscore-separated token -- always heat or cold."""
        return stem.split('_')[-1]

    def _strip_model(stem, model):
        """Strip '{model}_' prefix, raise clearly if it's missing."""
        prefix = f"{model}_"
        if not stem.startswith(prefix):
            raise ValueError(f"Stem {stem!r} doesn't start with expected prefix {prefix!r}")
        return stem[len(prefix):]

    def _expt_from_middle(middle):
        """
        Extract expt as the first underscore-separated token of `middle`.
        """
        return middle.split('_')[0]

    # --- thresholds: model -> member -> {ext_type: Dataset} ---
    if 'thresholds' in outputs:
        save_dir = base / 'thresholds'
        if save_dir.exists():
            data = {}
            for model_dir in sorted(save_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                for path in sorted(model_dir.glob("*.nc")):
                    stem = path.stem
                    try:
                        middle = _strip_model(stem, model)
                    except ValueError:
                        print(f"  Skipping (unexpected stem format): {path.name}")
                        continue
                    ext_type = _ext_type(stem)
                    # for thresholds, first token is member (no expt level)
                    member = middle.split('_')[0]
                    data.setdefault(model, {}).setdefault(member, {})[ext_type] = _open(path)
            result['thresholds'] = data

    # --- gwl_period_extremes[_matching]: model -> expt -> member -> Dataset ---
    for name in ['gwl_period_extremes', 'gwl_period_extremes_matching']:
        if name not in outputs:
            continue
        save_dir = base / name
        if not save_dir.exists():
            continue
        data = {}
        for model_dir in sorted(save_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for path in sorted(model_dir.glob("*.nc")):
                stem = path.stem
                try:
                    middle = _strip_model(stem, model)
                except ValueError:
                    print(f"  Skipping (unexpected stem format): {path.name}")
                    continue
                expt = _expt_from_middle(middle)
                m = _MEMBER_SUFFIX.search(middle)
                member = m.group('member') if m else None
                data.setdefault(model, {}).setdefault(expt, {})[member] = _open(path)
        result[name] = data

    # --- exceedances_*_median_multimodel: model -> expt -> {ext_type: Dataset} ---
    for name in ['exceedances_median_multimodel', 'exceedances_matching_median_multimodel']:
        if name not in outputs:
            continue
        save_dir = base / name
        if not save_dir.exists():
            continue
        data = {}
        for model_dir in sorted(save_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for path in sorted(model_dir.glob("*.nc")):
                stem = path.stem
                try:
                    middle = _strip_model(stem, model)
                except ValueError:
                    print(f"  Skipping (unexpected stem format): {path.name}")
                    continue
                ext_type = _ext_type(stem)
                expt = _expt_from_middle(middle)
                data.setdefault(model, {}).setdefault(expt, {})[ext_type] = _open(path)
        result[name] = data

    # --- branch_differences_*_median_multimodel: model -> expt -> {ext_type: Dataset} ---
    for name in ['branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel']:
        if name not in outputs:
            continue
        save_dir = base / name
        if not save_dir.exists():
            continue
        data = {}
        for model_dir in sorted(save_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for path in sorted(model_dir.glob("*.nc")):
                stem = path.stem
                try:
                    middle = _strip_model(stem, model)
                except ValueError:
                    print(f"  Skipping (unexpected stem format): {path.name}")
                    continue
                ext_type = _ext_type(stem)
                expt = _expt_from_middle(middle)
                data.setdefault(model, {}).setdefault(expt, {})[ext_type] = _open(path)
        result[name] = data

    # --- agreement[_matching]: expt -> {ext_type: Dataset} ---
    # No model-level directory here -- files sit directly in save_dir.
    # Expt is extracted as the first hyphen/non-underscore token in the filename.
    for name in ['agreement', 'agreement_matching']:
        if name not in outputs:
            continue
        save_dir = base / name
        if not save_dir.exists():
            continue
        data = {}
        for path in sorted(save_dir.glob("*.nc")):
            stem = path.stem
            ext_type = _ext_type(stem)
            # expt = first token (experiment names never contain underscores)
            expt = stem.split('_')[0]
            data.setdefault(expt, {})[ext_type] = _open(path)
        result[name] = data

    # apply model ordering to all model-keyed outputs
    if model_order is not None:
        model_keyed = [
            'thresholds',
            'gwl_period_extremes', 'gwl_period_extremes_matching',
            'exceedances_median_multimodel', 'exceedances_matching_median_multimodel',
            'branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel',
        ]
        for key in model_keyed:
            if key in result and result[key]:
                result[key] = utils.sort_models(result[key], model_order)

    return result


def load_land_ocean_masks(sftlf=None):
    if sftlf == None:
        sftlf = load_fx(['sftlf'])['sftlf'].sftlf
    land_mask = xr.where(sftlf > 50, 1, 0)   # 1 = land, 0 = ocean
    ocean_mask = xr.where(sftlf <= 50, 1, 0) # 1 = ocean, 0 = land
    return land_mask,ocean_mask

def load_amoc(activity_id):
    amoc_dir = PROC_DIR / activity_id / 'amoc'
    amoc_data = {model:{} for model in CDR_MODELS}
    for model in CDR_MODELS:
        path = amoc_dir / f"{model}_amoc_26N.nc"
        path_pi = amoc_dir / f"{model}_amoc_26N_piControl.nc"
        amoc = xr.open_dataarray(path).groupby('time.year').mean(dim='time')
        amoc_piControl = xr.open_dataarray(path_pi).groupby('time.year').mean(dim='time')
        amoc_data[model]['amoc_26N'] = amoc
        amoc_data[model]['amoc_piControl'] = amoc_piControl
        amoc_data[model]['anom'] =  amoc - amoc_piControl.mean(dim='year')
        amoc_data[model]['std_dev'] = amoc_piControl.std(dim='year')
    return amoc_data

def load_mrsos(models=CDR_MODELS):
    # load
    save_dir = RAW_DIR / 'mrsos'
    data = {model:{} for model in models}
    for expt in ['1pctCO2','1pctCO2-cdr']:
        expt_dir = save_dir / expt
        for model in models:
            path = expt_dir / f"{model}_{expt}_mrsos.nc"
            data[model][expt] = xr.open_dataarray(path,use_cftime=True)
    return data

def load_siconc(models=CDR_MODELS):
    # load
    save_dir = RAW_DIR / 'siconc'
    data = {model:{} for model in models}
    for expt in ['1pctCO2','1pctCO2-cdr']:
        expt_dir = save_dir / expt
        for model in models:
            path = expt_dir / f"{model}_{expt}_siconc.nc"
            data[model][expt] = xr.open_dataarray(path,use_cftime=True)
    return data

def load_areacello(models=CDR_MODELS):
    save_dir = RAW_DIR / 'Ofx'
    data = {}
    for model in models:
        path = save_dir / f"{model}_areacello.nc"
        data[model] = xr.open_dataarray(path)
    return data