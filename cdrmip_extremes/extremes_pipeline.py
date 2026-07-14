"""
extremes_pipeline.py
"""

from collections import defaultdict

import xarray as xr

from . import extremes
from . import stats
from . import utils
from . import save as save_io


def _map_leaves(nested_dict, func):
    out = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            out[k] = _map_leaves(v, func)
        else:
            out[k] = func(v)
    return out


class ExtremesPipeline:

    def __init__(self, regridded_picontrol, regridded_concat, crossing_years=None,gwl_matching=None,
                 varname='tas', gwl_window=21, ext_type_to_extrema=None):
        self.regridded_picontrol = regridded_picontrol
        self.regridded_concat = regridded_concat
        self.crossing_years = crossing_years or {}
        self.gwl_matching = gwl_matching or {}
        self.varname = varname
        self.gwl_window = gwl_window
        self.ext_type_to_extrema = ext_type_to_extrema or {'heat': 'max', 'cold': 'min'}

        self.extreme_months = {}       # model -> member -> Dataset (month of max/min)
        self.thresholds = {}           # model -> member -> {'heat': Dataset, 'cold': Dataset}
        self.gwl_period_extremes = {}          # model -> expt -> member -> DataArray (from crossing_years)
        self.gwl_period_extremes_matching = {} # model -> expt -> member -> DataArray (from gwl_matching)
        self.exceedances = {}                  # model -> expt -> {'heat': Dataset, 'cold': Dataset}
        self.exceedances_matching = {}
        self.branch_differences = {}           # model -> expt -> {'heat': DataArray, 'cold': DataArray}
        self.branch_differences_matching = {}
        self.agreement = {}                    # expt -> {'heat': DataArray, 'cold': DataArray}
        self.agreement_matching = {}
 
        # populated by compute_multi_model_stat() -- opt-in
        self.exceedances_median_multimodel = {}
        self.branch_differences_median_multimodel = {}
    # ------------------------------------------------------------------
    def compute_extreme_months(self):
        self.extreme_months = _map_leaves(
            self.regridded_picontrol,
            lambda ds: extremes.monthly_extrema(ds, varname=self.varname)
        )
        return self

    # ------------------------------------------------------------------
    def compute_thresholds(self):
        thresholds = defaultdict(dict)
        for model, member_dict in self.regridded_picontrol.items():
            for member, ds in member_dict.items():
                ext_months = self.extreme_months.get(model, {}).get(member)
                if ext_months is None or 'month' not in ext_months:
                    print(f"  No extreme-month data for {model}/{member} — skipping thresholds")
                    continue
                months = ext_months['month']
                max_mean = extremes.extreme_month_stat(ds, months.sel(extrema='max'), 'mean', self.varname)
                max_std = extremes.extreme_month_stat(ds, months.sel(extrema='max'), 'std', self.varname)
                min_mean = extremes.extreme_month_stat(ds, months.sel(extrema='min'), 'mean', self.varname)
                min_std = extremes.extreme_month_stat(ds, months.sel(extrema='min'), 'std', self.varname)
                thresholds[model][member] = {
                    'heat': extremes.heat_extreme_thresholds(max_mean, max_std),
                    'cold': extremes.cold_extreme_thresholds(min_mean, min_std),
                }
        self.thresholds = utils.defaultdict_to_dict(thresholds)
        return self

    # ------------------------------------------------------------------
    def _extract_gwl_period_extremes_for(self, gwl_source):
        """Internal helper: run the GWL-period extraction for one gwl-years
        source dict (either self.crossing_years or self.gwl_matching),
        returning a plain model -> expt -> member -> DataArray dict."""
        periods = defaultdict(lambda: defaultdict(dict))
        for model, expt_dict in self.regridded_concat.items():
            ext_months_by_member = self.extreme_months.get(model, {})
            for expt, member_dict in expt_dict.items():
                gwl_years_by_member = gwl_source.get(model, {}).get(expt)
                if gwl_years_by_member is None:
                    continue
                for member, ds in member_dict.items():
                    member_ext_months = ext_months_by_member.get(member)
                    gwl_years = (
                        gwl_years_by_member.get(member)
                        if isinstance(gwl_years_by_member, dict)
                        else gwl_years_by_member
                    )
                    if member_ext_months is None or gwl_years is None:
                        continue
                    ext_month_da = extremes.select_extreme_month(
                        ds, member_ext_months,
                        ext_type_to_extrema=self.ext_type_to_extrema,
                        varname=self.varname,
                    )
                    periods[model][expt][member] = stats.extract_gwl_period(
                        ext_month_da, gwl_years, window=self.gwl_window, time_dim='year'
                    )
        return utils.defaultdict_to_dict(periods)
 
    def extract_gwl_period_extremes(self):
        """
        Computes gwl_period_extremes from crossing_years (if provided), and
        gwl_period_extremes_matching from gwl_matching (if provided) --
        as two separate plain model -> expt -> member -> DataArray dicts,
        rather than overwriting one with the other.
        """
        if self.crossing_years:
            self.gwl_period_extremes = self._extract_gwl_period_extremes_for(self.crossing_years)
        else:
            print("  No crossing_years provided — skipping gwl_period_extremes")
 
        if self.gwl_matching:
            self.gwl_period_extremes_matching = self._extract_gwl_period_extremes_for(self.gwl_matching)
        else:
            print("  No gwl_matching provided — skipping gwl_period_extremes_matching")
 
        return self
        
    # ------------------------------------------------------------------

    def compute_exceedances(self, source_attr='gwl_period_extremes', save_as=None):
        """
        Per model/expt, member-mean exceedance frequency for each
        ext_type, at each gwl/branch.
 
        source_attr : 'gwl_period_extremes' (default, crossing_years-based)
            or 'gwl_period_extremes_matching' (gwl_matching-based).
        save_as : attribute name to store the result under. Defaults to
            'exceedances' for the default source_attr, or
            'exceedances_matching' if source_attr ends in '_matching'.
        """
        gwl_period_extremes = getattr(self, source_attr)
        if not gwl_period_extremes:
            print(f"  {source_attr} is empty — nothing to compute exceedances from")
            return self
 
        if save_as is None:
            save_as = 'exceedances_matching' if source_attr.endswith('_matching') else 'exceedances'
 

        drop_gwl = source_attr.endswith('_matching')
 
        exceedances = defaultdict(dict)
        for model, expt_dict in gwl_period_extremes.items():
            for expt, member_dict in expt_dict.items():
                member_exceedances = {ext_type: [] for ext_type in self.ext_type_to_extrema}
 
                for member, period_da in member_dict.items():
                    member_thresholds = self.thresholds.get(model, {}).get(member)
                    if member_thresholds is None:
                        print(f"  No thresholds for {model}/{member} — skipping exceedances")
                        continue
 
                    period_da = period_da.load()
                    for ext_type in self.ext_type_to_extrema:
                        exc = extremes.calculate_exceedances(
                            period_da, member_thresholds[ext_type], ext_type=ext_type
                        )
                        if drop_gwl and 'gwl' in exc.coords:
                            exc = exc.drop_vars('gwl')
                        member_exceedances[ext_type].append(exc.assign_coords(member=member))
 
                expt_result = {}
                for ext_type, exc_list in member_exceedances.items():
                    if not exc_list:
                        continue
                    expt_result[ext_type] = xr.concat(exc_list, dim='member').mean(dim='member')
                if expt_result:
                    exceedances[model][expt] = expt_result
 
        setattr(self, save_as, utils.defaultdict_to_dict(exceedances))
        return self
 
    # ------------------------------------------------------------------
    def compute_branch_differences(self, source_attr='exceedances', save_as=None, save_agreement_as=None):
        """
        For each ext_type and expt, ramp_down-minus-ramp_up difference
        per model, plus the multi-model agreement on its sign.
 
        source_attr : 'exceedances' (default) or 'exceedances_matching'.
        save_as / save_agreement_as : attribute names to store results
            under. Default to 'branch_differences'/'agreement', or their
            '_matching' counterparts if source_attr ends in '_matching'.
        """
        exceedances = getattr(self, source_attr)
        if not exceedances:
            print(f"  {source_attr} is empty — nothing to compute branch differences from")
            return self
 
        is_matching = source_attr.endswith('_matching')
        if save_as is None:
            save_as = 'branch_differences_matching' if is_matching else 'branch_differences'
        if save_agreement_as is None:
            save_agreement_as = 'agreement_matching' if is_matching else 'agreement'
        drop_coords = ['gwl'] if is_matching else None
 
        differences = defaultdict(dict)
        agreement = defaultdict(dict)
 
        for ext_type in self.ext_type_to_extrema:
            by_expt = defaultdict(dict)
            for model, expt_dict in exceedances.items():
                for expt, ext_dict in expt_dict.items():
                    if ext_type in ext_dict:
                        by_expt[expt][model] = ext_dict[ext_type]
 
            for expt, model_dict in by_expt.items():
                if not model_dict:
                    continue
                diffs, agree = extremes.calc_gwl_branch_differences(model_dict, drop_coords=drop_coords)
                for model, diff in diffs.items():
                    differences[model].setdefault(expt, {})[ext_type] = diff
                agreement.setdefault(expt, {})[ext_type] = agree
 
        setattr(self, save_as, utils.defaultdict_to_dict(differences))
        setattr(self, save_agreement_as, utils.defaultdict_to_dict(agreement))
        return self

    # ------------------------------------------------------------------
    # load exceedances into memory
    # ------------------------------------------------------------------
    def load_exceedances(self, attribute='exceedances'):
        """
        """
        def load_ext_dict(ext_dict):
            return {ext_type: ds.load() for ext_type, ds in ext_dict.items()}
 
        data = getattr(self, attribute)
        if not data:
            print(f"  {attribute} is empty — nothing to load")
            return self
 
        loaded = defaultdict(dict)
        for model, expt_dict in data.items():
            for expt, ext_dict in expt_dict.items():
                loaded[model][expt] = load_ext_dict(ext_dict)
        setattr(self, attribute, utils.defaultdict_to_dict(loaded))
        return self
 
    # ------------------------------------------------------------------
    def compute_multi_model_stat(self, attribute, reduction="median", expts=None, drop_coords=None):
        """
        drop_coords : list of str, optional
            Passed through to stats.reduce_models. Defaults to ['gwl']
            automatically when `attribute` is one of the gwl_matching-based
            outputs (ends in '_matching')
        """
        nested_dict = getattr(self, attribute)
        if not nested_dict:
            print(f"  {attribute} is empty — nothing to reduce")
            return self
 
        if drop_coords is None and attribute.endswith('_matching'):
            drop_coords = ['gwl']
 
        result = {}
        for ext_type in self.ext_type_to_extrema:
            per_ext_type = {
                model: {
                    expt: ext_dict[ext_type]
                    for expt, ext_dict in expt_dict.items()
                    if ext_type in ext_dict
                }
                for model, expt_dict in nested_dict.items()
            }
            per_ext_type = {m: e for m, e in per_ext_type.items() if e}
            if not per_ext_type:
                continue
 
            reduced = stats.reduce_models(per_ext_type, reduction=reduction, expts=expts,
                                           drop_coords=drop_coords)
            for model, expt_dict in reduced.items():
                for expt, ds in expt_dict.items():
                    result.setdefault(model, {}).setdefault(expt, {})[ext_type] = ds
 
        setattr(self, f"{attribute}_{reduction}_multimodel", result)
        return self
 
    # ------------------------------------------------------------------
    # Save outputs to disk.
    # ------------------------------------------------------------------
    def save(self, proc_dir, activity_id, category='extremes',overwrite=False,
              outputs=None):
        """
        Parameters
        ----------
        proc_dir : Path
        activity_id : str, e.g. 'CDRMIP'
        overwrite : bool
        outputs : list of str, optional
            Which outputs to save. Defaults to all of:
            ['thresholds',
             'gwl_period_extremes', 'gwl_period_extremes_matching',
             'exceedances_median_multimodel', 'exceedances_matching_median_multimodel',
             'branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel',
             'agreement', 'agreement_matching']
            Any attribute name on self that doesn't exist or is empty is
            skipped with a printed message rather than raising, so passing
            a name from the crossing_years-only set when you only computed
            gwl_matching (or a typo) won't silently do nothing without
            telling you.
        """
        from pathlib import Path
        base = Path(proc_dir) / activity_id / category
        defaults = [
            'thresholds',
            'gwl_period_extremes', 'gwl_period_extremes_matching',
            'exceedances_median_multimodel', 'exceedances_matching_median_multimodel',
            'branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel',
            'agreement', 'agreement_matching',
        ]
        outputs = outputs or defaults
 
        def _ds(obj):
            """Coerce DataArray to Dataset for saving."""
            return obj.to_dataset() if isinstance(obj, xr.DataArray) else obj
 
        def _get(name):
            data = getattr(self, name, None)
            if not data:
                if name in outputs:
                    print(f"  Skipping '{name}': empty or not computed")
                return None
            return data
 
        # --- thresholds: model -> member -> {ext_type: Dataset} ---
        if 'thresholds' in outputs:
            data = _get('thresholds')
            if data:
                save_dir = base / 'thresholds'
                for model, member_dict in data.items():
                    model_dir = save_dir / model
                    model_dir.mkdir(parents=True, exist_ok=True)
                    for member, ext_dict in member_dict.items():
                        for ext_type, ds in ext_dict.items():
                            path = model_dir / f"{model}_{member}_{self.varname}_thresholds_{ext_type}.nc"
                            save_io._save_nc(_ds(ds), path, overwrite=overwrite)
 
        # --- gwl_period_extremes[_matching]: model -> expt -> member -> DataArray ---
        for name in ['gwl_period_extremes', 'gwl_period_extremes_matching']:
            if name not in outputs:
                continue
            data = _get(name)
            if not data:
                continue
            save_dir = base / name
            for model, expt_dict in data.items():
                model_dir = save_dir / model
                model_dir.mkdir(parents=True, exist_ok=True)
                for expt, member_dict in expt_dict.items():
                    for member, da in member_dict.items():
                        path = model_dir / f"{model}_{expt}_{member}_{self.varname}_{name}.nc"
                        save_io._save_nc(_ds(da), path, overwrite=overwrite)
 
        # --- exceedances_median_multimodel[_matching]: model -> expt -> {ext_type: Dataset} ---
        for name in ['exceedances_median_multimodel', 'exceedances_matching_median_multimodel']:
            if name not in outputs:
                continue
            data = _get(name)
            if not data:
                continue
            save_dir = base / name
            for model, expt_dict in data.items():
                model_dir = save_dir / model
                model_dir.mkdir(parents=True, exist_ok=True)
                for expt, ext_dict in expt_dict.items():
                    for ext_type, ds in ext_dict.items():
                        path = model_dir / f"{model}_{expt}_{self.varname}_{name}_{ext_type}.nc"
                        save_io._save_nc(_ds(ds), path, overwrite=overwrite)
 
        # --- branch_differences_median_multimodel[_matching]: model -> expt -> {ext_type: DataArray} ---
        for name in ['branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel']:
            if name not in outputs:
                continue
            data = _get(name)
            if not data:
                continue
            save_dir = base / name
            for model, expt_dict in data.items():
                model_dir = save_dir / model
                model_dir.mkdir(parents=True, exist_ok=True)
                for expt, ext_dict in expt_dict.items():
                    for ext_type, da in ext_dict.items():
                        path = model_dir / f"{model}_{expt}_{self.varname}_{name}_{ext_type}.nc"
                        save_io._save_nc(_ds(da), path, overwrite=overwrite)
 
        # --- agreement[_matching]: expt -> {ext_type: DataArray} ---
        for name in ['agreement', 'agreement_matching']:
            if name not in outputs:
                continue
            data = _get(name)
            if not data:
                continue
            save_dir = base / name
            save_dir.mkdir(parents=True, exist_ok=True)
            for expt, ext_dict in data.items():
                for ext_type, da in ext_dict.items():
                    path = save_dir / f"{expt}_{self.varname}_{name}_{ext_type}.nc"
                    save_io._save_nc(_ds(da), path, overwrite=overwrite)
 
        return self
 
    # ------------------------------------------------------------------
    def run(self):
        """
        Runs the full pipeline. Computes both the crossing_years-based
        and gwl_matching-based outputs, for whichever of the two were
        actually provided to __init__ (compute_exceedances/
        compute_branch_differences print a message and skip cleanly if
        the corresponding *_period_extremes attribute is empty).
        """
        self.compute_extreme_months().compute_thresholds().extract_gwl_period_extremes()
 
        if self.gwl_period_extremes:
            self.compute_exceedances(source_attr='gwl_period_extremes')
            self.compute_branch_differences(source_attr='exceedances')
 
        if self.gwl_period_extremes_matching:
            self.compute_exceedances(source_attr='gwl_period_extremes_matching')
            self.compute_branch_differences(source_attr='exceedances_matching')
 
        return self