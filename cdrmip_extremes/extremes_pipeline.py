"""
extremes_pipeline.py

Module to expedite the hottest/coldest month temperature analysis.
This module makes use of the output of load_pipeline.py and draws from the functions defined
in extremes.py to identify the pre-industrial hottest/coldest months for each grid-cell,
define 'extreme' thresholds based on the mean and standard deviation temperatures for these months
and then determine how frequently these thresholds are exceeded.
"""

from collections import defaultdict
import xarray as xr
from pathlib import Path

from . import extremes, stats, utils, gwl, io_disk


class ExtremesPipeline:

    def __init__(self, regridded_piControl, regridded_concat, crossing_years=None,
                 gwl_matching=None, varname='tas', gwl_window=21, ext_type_to_extrema=None):
        
        self.regridded_piControl = regridded_piControl
        self.regridded_concat = regridded_concat
        self.crossing_years = crossing_years or {}
        self.gwl_matching = gwl_matching or {}
        self.varname = varname
        self.gwl_window = gwl_window
        # config to convert min/max coordinates to 'cold'/'heat'
        self.ext_type_to_extrema = ext_type_to_extrema or {'heat': 'max', 'cold': 'min'}

        self.extreme_months = {}       
        self.thresholds = {}           
        self.gwl_period_extremes = {}          
        self.gwl_period_extremes_matching = {} 
        self.exceedances = {}                  
        self.exceedances_matching = {}
        self.branch_differences = {}           
        self.branch_differences_matching = {}
        self.agreement = {}                    
        self.agreement_matching = {}


        self.exceedances_median_multimodel = {}
        self.branch_differences_median_multimodel = {}

    # ------------------------------------------------------------------
    def compute_extreme_months(self):
        self.extreme_months = utils.map_nested(
            lambda ds: extremes.monthly_extrema(ds, varname=self.varname),
            self.regridded_piControl,
        )
        return self

    # ------------------------------------------------------------------
    def compute_thresholds(self):
        thresholds = defaultdict(dict)
        for model, member_dict in self.regridded_piControl.items():
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
        """
        """
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
                    periods[model][expt][member] = gwl.extract_gwl_period(
                        ext_month_da, gwl_years, window=self.gwl_window, time_dim='year'
                    )
        return utils.defaultdict_to_dict(periods)
 
    def extract_gwl_period_extremes(self):
        """
        Computes gwl_period_extremes from crossing_years (if provided), and
        gwl_period_extremes_matching from gwl_matching (if provided) 
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

    def compute_branch_differences(self, source_attr='exceedances', percent=False):
        save_as = 'branch_differences' + ('_matching' if source_attr.endswith('_matching') else '')
        diffs = utils.map_nested(
            lambda exc: gwl.gwl_branch_difference(exc, percent=percent),
            getattr(self, source_attr),
        )
        setattr(self, save_as, diffs)
        return self
    
    # ------------------------------------------------------------------
    
    def compute_agreement(self, source_attr='branch_differences', threshold=0.75):
        save_as = 'agreement' + ('_matching' if source_attr.endswith('_matching') else '')
    
        by_expt = defaultdict(lambda: defaultdict(dict))
        for model, expt_dict in getattr(self, source_attr).items():
            for expt, ext_dict in expt_dict.items():
                for ext_type, diff in ext_dict.items():
                    by_expt[expt][ext_type][model] = diff
    
        setattr(self, save_as, {
            expt: {ext: stats.calc_agreement(models, threshold) for ext, models in ext_dict.items()}
            for expt, ext_dict in by_expt.items()
        })
        return self

    # ------------------------------------------------------------------
    def load(self, attribute='exceedances'):
        setattr(self, attribute, utils.map_nested(lambda x: x.load(), getattr(self, attribute)))
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
 
            reduced = utils.reduce_models(per_ext_type, reduction=reduction, expts=expts,
                                           drop_coords=drop_coords)
            for model, expt_dict in reduced.items():
                for expt, ds in expt_dict.items():
                    result.setdefault(model, {}).setdefault(expt, {})[ext_type] = ds
 
        setattr(self, f"{attribute}_{reduction}_multimodel", result)
        return self

    # ------------------------------------------------------------------
    def run(self):
        """
        Runs the full pipeline.
        """
        self.compute_extreme_months().compute_thresholds().extract_gwl_period_extremes()
 
        if self.gwl_period_extremes:
            self.compute_exceedances(source_attr='gwl_period_extremes')
            self.compute_branch_differences(source_attr='exceedances')
            self.compute_agreement(source_attr='branch_differences')
 
        if self.gwl_period_extremes_matching:
            self.compute_exceedances(source_attr='gwl_period_extremes_matching')
            self.compute_branch_differences(source_attr='exceedances_matching')
            self.compute_agreement(source_attr='branch_differences')
 
        return self

    def save(self, proc_dir, activity_id, category='extremes', overwrite=False, outputs=None):
        outputs = outputs or [
            'thresholds',
            'gwl_period_extremes', 'gwl_period_extremes_matching',
            'exceedances_median_multimodel', 'exceedances_matching_median_multimodel',
            'branch_differences_median_multimodel', 'branch_differences_matching_median_multimodel',
            'agreement', 'agreement_matching',
        ]
        base = Path(proc_dir) / activity_id / category
        for name in outputs:
            data = getattr(self, name, None)
            if not data:
                print(f"  Skipping '{name}': empty or not computed")
                continue
            io_disk.save_nested(data, base / name, f"{self.varname}_{name}", overwrite=overwrite)
        return self
 
