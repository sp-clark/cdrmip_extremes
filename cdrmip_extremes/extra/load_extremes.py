"""
load_extremes -- loads outputs saved by ExtremesPipeline.save().

File layout (must match ExtremesPipeline.save() exactly):
    proc_dir/activity_id/{varname}/
        thresholds/                                  {model}/{model}_{member}_{varname}_thresholds_{ext_type}.nc
        gwl_period_extremes/                          {model}/{model}_{expt}_{member}_{varname}_gwl_period_extremes.nc
        gwl_period_extremes_matching/                 {model}/{model}_{expt}_{member}_{varname}_gwl_period_extremes_matching.nc
        exceedances_median_multimodel/                {model}/{model}_{expt}_{varname}_exceedances_median_multimodel_{ext_type}.nc
        exceedances_matching_median_multimodel/       {model}/{model}_{expt}_{varname}_exceedances_matching_median_multimodel_{ext_type}.nc
        branch_differences_median_multimodel/         {model}/{model}_{expt}_{varname}_branch_differences_median_multimodel_{ext_type}.nc
        branch_differences_matching_median_multimodel/{model}/{model}_{expt}_{varname}_branch_differences_matching_median_multimodel_{ext_type}.nc
        agreement/                                    {expt}_{varname}_agreement_{ext_type}.nc
        agreement_matching/                           {expt}_{varname}_agreement_matching_{ext_type}.nc

Note: this supersedes any earlier load_extremes you may have -- the
filename pattern changed (the category name, e.g. 'exceedances_median_multimodel',
is now embedded in the filename itself rather than just 'exceedances') so
the *_matching outputs never collide with the non-matching ones in the
same folder.
"""

import re
from pathlib import Path

import xarray as xr

from . import utils

_MEMBER_SUFFIX = re.compile(r'_(?P<member>r\d+i\d+p\d+f\d+)$')

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
    Load outputs saved by ExtremesPipeline.save(), reconstructing the same
    nested-dict structure that ExtremesPipeline produces.

    category : str, default 'extremes'
        Top-level subfolder under proc_dir/activity_id/ -- must match
        whatever `category` was passed to ExtremesPipeline.save().
    outputs : list of str, optional
        Which categories to load. Defaults to all nine in DEFAULT_OUTPUTS.
        Any category whose folder doesn't exist is skipped silently.
    model_order : list of str, optional
        Applied via utils.sort_models to every model-keyed output.

    Returns
    -------
    dict {output_name: nested_dict}, only for outputs actually found on disk.
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
        Safe because CMIP6 experiment names use hyphens, never underscores,
        so the expt portion always ends at the first '_'. Everything after
        (varname fragments, short/long category names) is ignored -- this
        makes the loader robust to whatever naming convention was used when
        saving (old shortened names, new full names, with or without varname).
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