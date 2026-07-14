"""
pipeline.py

"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Union

import xarray as xr

from .load_raw import search_and_load, search_and_load_parent
from .processing import concatenate_branches, regrid_to_common
from . import utils, stats
from .configs import DEFAULT_CHUNKS, DEFAULT_VARS_TO_DROP
from . import save as save_io


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for one top-level experiment to load and process."""
    experiment_id: str
    table_id: str
    variable_id: str
    grid_label: str = 'gn'
    member_id: Optional[Union[str, list]] = None     # None -> take all available members
    anomaly_reference: str = 'period'                 # 'period' or 'picontrol'
    reference_period: Optional[tuple] = (1850, 1900)# used only if anomaly_reference == 'period'
    anomaly_time_resolution: str = 'annual'
    stop_concat_before: tuple = ('piControl',)        # parent experiments at which to stop chaining
    
    # --- GWL crossing-year analysis (compute_crossing_years) ---
    rolling_window: int = 21
    crossing_gwls: Optional[list] = None        # e.g. [1.5, 2.0, 3.0]; None -> skip for this experiment
    overshoot: bool = True
    gwl_extract_window: Optional[int] = None    # window for extract_gwl_periods; None -> use rolling_window

    # Skip loading a member entirely if its time series doesn't extend
    # beyond `threshold` (passed straight through to search_and_load).
    # e.g. only load ssp534-over members that run beyond 2101.
    check_end_year: bool = False
    threshold: Optional[int] = None
 
    def __post_init__(self):
        if self.check_end_year and self.threshold is None:
            raise ValueError(
                f"{self.experiment_id}: check_end_year=True requires threshold to be set"
            )
 
    # --- GWL matching analysis (compute_gwl_matching) ---
    gwl_match_period: Optional[tuple] = None        # e.g. (320, 341) or (2170, 2201); None -> skip
    gwl_match_search_period: Optional[tuple] = None  # None -> search pre-peak automatically


@dataclass
class PipelineConfig:
    experiments: list  # list[ExperimentConfig]
    chunks: Optional[dict] = None
    vars_to_drop: Optional[list] = None

    # Branch-date overrides
    branch_date_overrides: dict = field(default_factory=dict)

    realign_years_overrides: dict = field(default_factory=dict)

    # {source_id: parent_source_id}, for models whose 'parent_source_id'
    # attribute is missing/unreliable.
    source_id_overrides: Optional[dict] = None


# ---------------------------------------------------------------------------
# Generic helper: apply a function to every leaf Dataset in an arbitrarily
# nested dict (works whether the dict is model/member or model/expt/member,
# or any other depth).
# ---------------------------------------------------------------------------

def _map_leaves(nested_dict, func):
    out = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            out[k] = _map_leaves(v, func)
        else:
            out[k] = func(v)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CDRMIPPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.chunks = DEFAULT_CHUNKS if self.config.chunks is None else self.config.chunks
        self.config.vars_to_drop = (
            DEFAULT_VARS_TO_DROP if self.config.vars_to_drop is None else self.config.vars_to_drop
        )

        # Shared across every search_and_load_parent call (chasing parents
        # and loading piControl both go through this), so an identical
        # parent dataset is only searched for / opened once.
        self._parent_cache = {}

        # raw[experiment_id][model][member] = ds
        self.raw = {}

        # concat[model][experiment_id][member] = ds   (chained to parents)
        self.concat = {}

        # picontrol[model][member] = ds
        self.picontrol = {}

        # populated by regrid()
        self.regridded_concat = {}
        self.regridded_picontrol = {}

        # populated by compute_anomalies() / compute_global_mean() / compute_gsat_rolling()
        self.anomaly = {}
        self.gsat = {}
        self.gsat_rolling = {}
        self.peak_warming = {}

        # populated by compute_crossing_years() / compute_gwl_matching() — opt-in,
        self.crossing_years = {}
        self.gwl_matching = {}
        self.gwl_differences = {}
        self.gwl_periods = {}
        self.gwl_reversibility = {}

        # populated by compute_ensemble_summaries() — opt-in
        self.gsat_median = {}
        self.gsat_rolling_median = {}
        self.anomaly_median = {}
        self.crossing_years_median = {}
        self.gwl_differences_median = {}
        self.gwl_reversibility_median = {}
        self.gwl_differences_median_multimodel = {}
        self.gwl_reversibility_median_multimodel = {}

    # ------------------------------------------------------------------
    # Step 1: search & load raw data for each configured experiment
    # ------------------------------------------------------------------
    def load_raw(self):
        for cfg in self.config.experiments:
            print(f"=== Loading {cfg.experiment_id} ===")
            self.raw[cfg.experiment_id] = search_and_load(
                nest_by=('source_id', 'member_id'),
                experiment_id=cfg.experiment_id,
                table_id=cfg.table_id,
                variable_id=cfg.variable_id,
                grid_label=cfg.grid_label,
                member_id=cfg.member_id,
                chunks=self.config.chunks,
                vars_to_drop=self.config.vars_to_drop,
                check_end_year=cfg.check_end_year,
                threshold=cfg.threshold,
            )
        return self

    # ------------------------------------------------------------------
    # Step 2: chase the parent chain
    # ------------------------------------------------------------------
    def _lookup_branch_date(self, source_id, experiment_id, parent_experiment_id):
        overrides = self.config.branch_date_overrides
        if (source_id, experiment_id) in overrides:
            return overrides[(source_id, experiment_id)]
        if parent_experiment_id in overrides:
            return overrides[parent_experiment_id]
        return None  # fall back to branch_time_in_parent metadata

    def _lookup_realign_years(self, source_id, experiment_id, parent_experiment_id):
        overrides = self.config.realign_years_overrides
        if (source_id, experiment_id) in overrides:
            return overrides[(source_id, experiment_id)]
        if parent_experiment_id in overrides:
            return overrides[parent_experiment_id]
        return None

    def _chase_parents(self, ds, stop_before):
        """
        Repeatedly find-and-concatenate to the parent experiment until the
        parent_experiment_id is one of `stop_before` (typically 'piControl')
        or no parent metadata is found.

        Note: after each concatenation, ds.attrs reflects the *parent's*
        attrs (xr.concat default combine_attrs='override' takes attrs from
        the first dataset in the list, which is parent_ds).
        """
        while True:
            parent_experiment_id = ds.attrs.get('parent_experiment_id')
            if not parent_experiment_id or parent_experiment_id in stop_before:
                return ds

            source_id = ds.attrs['source_id']
            experiment_id = ds.attrs['experiment_id']

            parent_ds = search_and_load_parent(
                ds,
                chunks=self.config.chunks,
                vars_to_drop=self.config.vars_to_drop,
                source_id_overrides=self.config.source_id_overrides,
                cache=self._parent_cache,
            )
            if parent_ds is None:
                return ds

            realign_years = self._lookup_realign_years(
                source_id, experiment_id, parent_experiment_id
            )
            if realign_years is not None:
                print(
                    f"Concatenating {experiment_id} -> {parent_experiment_id} "
                    f"[{source_id}] via realign_years={realign_years}"
                )
                ds = concatenate_branches(
                    ds, parent_ds, time_dim='time', realign_years=realign_years
                )
            else:
                branch_date = self._lookup_branch_date(
                    source_id, experiment_id, parent_experiment_id
                )
                print(
                    f"Concatenating {experiment_id} -> {parent_experiment_id} [{source_id}]"
                    + (f" (branch_date override: {branch_date})" if branch_date else "")
                )
                ds = concatenate_branches(
                    ds, parent_ds, time_dim='time', branch_date=branch_date
                )

    def concatenate_to_parents(self):
        concat = defaultdict(lambda: defaultdict(dict))
        for cfg in self.config.experiments:
            for model, member_dict in self.raw[cfg.experiment_id].items():
                for member, ds in member_dict.items():
                    full_ds = self._chase_parents(ds, stop_before=cfg.stop_concat_before)
                    # Re-stamp experiment_id since attrs now reflect whichever
                    # ancestor the chain stopped at, not the original experiment.
                    full_ds = full_ds.assign_attrs(experiment_id=cfg.experiment_id)
                    concat[model][cfg.experiment_id][member] = full_ds
        self.concat = utils.defaultdict_to_dict(concat)
        return self

    # ------------------------------------------------------------------
    # Step 3: load piControl (deduplicated across experiments per model/member)
    # ------------------------------------------------------------------
    def load_picontrol(self):
        picontrol = defaultdict(dict)
        for model, expt_dict in self.concat.items():
            for expt, member_dict in expt_dict.items():
                for member, ds in member_dict.items():
                    parent_ds = search_and_load_parent(
                        ds,
                        chunks=self.config.chunks,
                        vars_to_drop=self.config.vars_to_drop,
                        source_id_overrides=self.config.source_id_overrides,
                        cache=self._parent_cache,
                    )
                    if parent_ds is None:
                        continue
                    if member not in picontrol[model]:
                        picontrol[model][member] = parent_ds
                    elif not picontrol[model][member].identical(parent_ds):
                        raise ValueError(f"Inconsistent piControl for {model} {member}")
        self.picontrol = utils.defaultdict_to_dict(picontrol)
        return self

    # ------------------------------------------------------------------
    # Step 4: regrid
    # ------------------------------------------------------------------
    def regrid(self):
        def regrid_ds(ds):
            if 'tas' in ds:
                return regrid_to_common(ds)
            return ds

        self.regridded_concat = _map_leaves(self.concat, regrid_ds)
        self.regridded_picontrol = _map_leaves(self.picontrol, regrid_ds)
        return self

    # ------------------------------------------------------------------
    # Step 5: anomalies 
    # ------------------------------------------------------------------
    def compute_anomalies(self):
        cfg_by_expt = {c.experiment_id: c for c in self.config.experiments}
        anomaly = defaultdict(lambda: defaultdict(dict))
 
        for model, expt_dict in self.regridded_concat.items():
            for expt, member_dict in expt_dict.items():
                cfg = cfg_by_expt[expt]
                for member, ds in member_dict.items():
                    if 'tas' not in ds:
                        anomaly[model][expt][member] = ds
                        continue
 
                    native = cfg.anomaly_time_resolution == 'native'
                    ds_annual = ds.groupby('time.year').mean(dim='time')
                    # target series the reference climatology gets subtracted from:
                    # native time coordinate if requested, else the annual means.
                    ds_target = ds if native else ds_annual
 
                    if cfg.anomaly_reference == 'period':
                        if native:
                            reference = ds_annual.sel(
                                year=slice(*cfg.reference_period)
                            ).mean('year')
                            anomaly[model][expt][member] = ds_target - reference
                        else:
                            anomaly[model][expt][member] = utils.calc_anomaly(
                                ds_annual, ds_annual,
                                slice(*cfg.reference_period), time_dim='year'
                            )
 
                    elif cfg.anomaly_reference == 'picontrol':
                        pi_ds = self.regridded_picontrol.get(model, {}).get(member)
                        if pi_ds is None:
                            print(f"  No piControl for {model}/{member} — skipping anomaly")
                            continue
                        pi_annual = pi_ds.groupby('time.year').mean(dim='time')
                        reference = pi_annual.mean('year')
                        anomaly[model][expt][member] = ds_target - reference
 
                    else:
                        raise ValueError(
                            f"Unknown anomaly_reference '{cfg.anomaly_reference}' for {expt}"
                        )
 
        self.anomaly = utils.defaultdict_to_dict(anomaly)
        return self

    # ------------------------------------------------------------------
    # Step 6: global mean
    # ------------------------------------------------------------------
    def compute_global_mean(self):
        def global_mean_ds(ds):
            if 'tas' not in ds:
                return ds
            return utils.global_mean(ds).to_dataset(name='gsat')

        self.gsat = _map_leaves(self.anomaly, global_mean_ds)
        return self

    # ------------------------------------------------------------------
    # Optional: force computation of gsat
    # ------------------------------------------------------------------
    def load_gsat(self):
        self.gsat = _map_leaves(self.gsat, lambda ds: ds.load())
        return self
        
    # ------------------------------------------------------------------
    # Step 6b: gsat_rolling
    # ------------------------------------------------------------------
    def compute_gsat_rolling(self):
        cfg_by_expt = {c.experiment_id: c for c in self.config.experiments}
        rolling = defaultdict(lambda: defaultdict(dict))
    
        for model, expt_dict in self.gsat.items():
            for expt, member_dict in expt_dict.items():
                cfg = cfg_by_expt[expt]
                time_dim = 'time' if cfg.anomaly_time_resolution == 'native' else 'year'
                for member, ds in member_dict.items():
                    if 'gsat' not in ds:
                        continue
                    rolling[model][expt][member] = stats._rolling_mean(
                        ds['gsat'], window=cfg.rolling_window, time_dim=time_dim
                    )
    
        self.gsat_rolling = utils.defaultdict_to_dict(rolling)
        return self

    # ------------------------------------------------------------------
    # Step 6c (optional): peak warming value + year, from gsat_rolling.
    # ------------------------------------------------------------------
    def compute_peak_warming(self):
        def peak_fn(da):
            peak, peak_year = stats.peak_warming(da, time_dim='year')
            return {'peak': peak, 'peak_year': peak_year}
 
        self.peak_warming = _map_leaves(self.gsat_rolling, peak_fn)
        return self
    # ------------------------------------------------------------------
    # Step 7 (optional): GWL crossing years -- for each configured
    # experiment with crossing_gwls set, find the year(s) the rolling
    # GSAT crosses each threshold.
    # ------------------------------------------------------------------
    def compute_crossing_years(self):
        cfg_by_expt = {c.experiment_id: c for c in self.config.experiments}
        crossing_years = defaultdict(lambda: defaultdict(dict))
 
        for model, expt_dict in self.gsat.items():
            for expt, member_dict in expt_dict.items():
                cfg = cfg_by_expt[expt]
                if not cfg.crossing_gwls:
                    continue
                for member, ds in member_dict.items():
                    if 'gsat' not in ds:
                        continue
                    crossing_years[model][expt][member] = stats.find_crossing_years(
                        ds['gsat'], window=cfg.rolling_window,
                        gwls=cfg.crossing_gwls, time_dim='year',
                        overshoot=cfg.overshoot,
                    )
 
        self.crossing_years = utils.defaultdict_to_dict(crossing_years)
        return self

    # ------------------------------------------------------------------
    # Step 7b (optional): extract multi-year windows of the (non-gsat)
    # anomaly field centred on each GWL crossing year found above. Needs
    # compute_crossing_years() to have been run first for the relevant
    # experiment(s).
    # ------------------------------------------------------------------
    def extract_gwl_periods(self):
        cfg_by_expt = {c.experiment_id: c for c in self.config.experiments}
        periods = defaultdict(lambda: defaultdict(dict))
        reversibility = defaultdict(lambda: defaultdict(dict))
 
        for model, expt_dict in self.anomaly.items():
            for expt, member_dict in expt_dict.items():
                cfg = cfg_by_expt[expt]
                if not cfg.crossing_gwls:
                    continue
                gwl_years_by_member = self.crossing_years.get(model, {}).get(expt, {})
                window = cfg.gwl_extract_window or cfg.rolling_window
 
                for member, ds in member_dict.items():
                    if 'tas' not in ds:
                        continue
                    gwl_years = gwl_years_by_member.get(member)
                    if gwl_years is None:
                        continue
                    period_da = stats.extract_gwl_period(
                        ds['tas'], gwl_years, window=window, time_dim='year'
                    )
                    periods[model][expt][member] = period_da
 
                    if cfg.overshoot:
                        reversibility[model][expt][member] = stats.gwl_branch_difference(period_da)
 
        self.gwl_periods = utils.defaultdict_to_dict(periods)
        self.gwl_reversibility = utils.defaultdict_to_dict(reversibility)
        return self
 
    # ------------------------------------------------------------------
    # Step 8 (optional): GWL matching -- for each configured experiment
    # with gwl_match_period set, find the GSAT level characterising that
    # period and the earlier year that reached the same level, plus the
    # anomaly difference between the two periods
    # ------------------------------------------------------------------
    def compute_gwl_matching(self):
        cfg_by_expt = {c.experiment_id: c for c in self.config.experiments}
        matching = defaultdict(lambda: defaultdict(dict))
        differences = defaultdict(lambda: defaultdict(dict))
 
        for model, expt_dict in self.gsat.items():
            for expt, member_dict in expt_dict.items():
                cfg = cfg_by_expt[expt]
                if cfg.gwl_match_period is None:
                    continue
                for member, ds in member_dict.items():
                    if 'gsat' not in ds:
                        continue
 
                    search_slice = (
                        slice(*cfg.gwl_match_search_period)
                        if cfg.gwl_match_search_period else None
                    )
                    # gwl, matching_year = stats.find_matching_gwl_year(
                    #     ds['gsat'], window=cfg.rolling_window,
                    #     match_slice=slice(*cfg.gwl_match_period),
                    #     time_dim='year', search_slice=search_slice,
                    # )
                    # matching[model][expt][member] = {'gwl': gwl, 'matching_year': matching_year}
                    matching_da = stats.find_matching_gwl_year(
                        ds['gsat'], window=cfg.rolling_window,
                        match_slice=slice(*cfg.gwl_match_period),
                        time_dim='year', search_slice=search_slice,
                    )
                    matching[model][expt][member] = matching_da
                    
 
                    anom_ds = self.anomaly.get(model, {}).get(expt, {}).get(member)
                    if anom_ds is not None and 'tas' in anom_ds:
                        matching_year = int(matching_da.isel(gwl=0).sel(branch='ramp_up').values)
                        
                        differences[model][expt][member] = stats.gwl_matched_anomaly_difference(
                            anom_ds['tas'], slice(*cfg.gwl_match_period), matching_year,
                            time_dim='year',
                        )
 
        self.gwl_matching = utils.defaultdict_to_dict(matching)
        self.gwl_differences = utils.defaultdict_to_dict(differences)
        return self

    # ------------------------------------------------------------------
    # General-purpose utility: relabel a tree's time axes onto a common
    # start year.
    # ------------------------------------------------------------------
    def realign_time(self, nested_dict, start_year=1):
        def shift(ds):
            return utils.shift_time_to_start_year(ds, start_year=start_year)
        return _map_leaves(nested_dict, shift)

    # ------------------------------------------------------------------
    # Optional: build a DataTree at the end, purely for saving/plotting
    # convenience — not used internally by the compute steps above.
    # ------------------------------------------------------------------
    def to_datatree(self, nested_dict):
        flat = {}
        def _walk(d, path):
            for k, v in d.items():
                if isinstance(v, dict):
                    _walk(v, path + [k])
                else:
                    flat['/' + '/'.join(path + [k])] = v
        _walk(nested_dict, [])
        return xr.DataTree.from_dict(flat)

    # ------------------------------------------------------------------
    # Optional: calculate median/mean across ensemble members for experiment
    # ------------------------------------------------------------------
    def compute_member_ensemble(self, attribute, reduction="median", save_as=None,
                                 drop_coords=None):
        """
        drop_coords : list of str, optional
            Passed to stats.reduce_members. Defaults to ['gwl'] automatically
            for attributes whose name contains 'matching' or 'differences' or
            'reversibility', since those carry per-member gwl values from the
            gwl_matching approach that must not be used as alignment keys
            during the member-concat. Pass [] explicitly to suppress this.
        """
        nested_dict = getattr(self, attribute)
 
        # auto-drop gwl for matching-derived attributes
        if drop_coords is None:
            if any(k in attribute for k in ('matching', 'differences', 'reversibility')):
                drop_coords = ['gwl']
 
        output = defaultdict(dict)
 
        for model, expt_dt in nested_dict.items():
            for expt, member_dt in expt_dt.items():
                output[model][expt] = stats.reduce_members(
                    member_dt,
                    reduction=reduction,
                    drop_coords=drop_coords,
                )
 
        if save_as is not None:
            setattr(self, save_as, dict(output))
 
        return dict(output)
    # ------------------------------------------------------------------
    # Optional: calculate multi-model median
    # ------------------------------------------------------------------
    def compute_multi_model_stat(
        self,
        attribute,
        reduction="median",
        save_as=None,
        expts=None,
        drop_coords=None,
    ):
        """
        Compute a multi-model statistic for a pipeline attribute.
 
        Parameters
        ----------
        attribute : str
            Name of a pipeline attribute with structure
            {model: {experiment: xr.Dataset}}.
        reduction : str
            Reduction to perform (e.g. 'median', 'mean', 'std').
        save_as : str, optional
            Name of the pipeline attribute in which to store the result.
        expts : iterable, optional
            Experiments to include.
        drop_coords : list of str, optional
            Passed to stats.reduce_models. Defaults to ['gwl'] automatically
            for attributes whose name contains 'matching', 'differences', or
            'reversibility' -- same reasoning as compute_member_ensemble.
        """
        nested_dict = getattr(self, attribute)
 
        if drop_coords is None:
            if any(k in attribute for k in ('matching', 'differences', 'reversibility')):
                drop_coords = ['gwl']
 
        result = stats.reduce_models(
            nested_dict,
            reduction=reduction,
            expts=expts,
            drop_coords=drop_coords,
        )
 
        if save_as is not None:
            setattr(self, save_as, result)
 
        return result

    # ------------------------------------------------------------------
    # Optional wrapper: runs compute_member_ensemble() for
    # gsat, gsat_rolling, anomaly, crossing_years, gwl_differences, and
    # gwl_reversibility, then compute_multi_model_stat() on top of the
    # ensemble-reduced gwl_differences/gwl_reversibility.
    # ------------------------------------------------------------------
    def compute_ensemble_summaries(self, member_reduction="median", model_reduction="median"):
        member_reduce_attrs = [
            'gsat', 'gsat_rolling', 'anomaly','gwl_matching',
            'crossing_years', 'gwl_differences', 'gwl_reversibility',
        ]
        for attr in member_reduce_attrs:
            data = getattr(self, attr)
            if not data:
                print(f"  Skipping {attr}: empty (not yet computed)")
                continue
            self.compute_member_ensemble(
                attr, reduction=member_reduction, save_as=f"{attr}_{member_reduction}"
            )
 
        model_reduce_attrs = ['gwl_differences', 'gwl_reversibility']
        for attr in model_reduce_attrs:
            member_reduced_attr = f"{attr}_{member_reduction}"
            data = getattr(self, member_reduced_attr, None)
            if not data:
                print(f"  Skipping multi-model {attr}: {member_reduced_attr} empty/not computed")
                continue
            self.compute_multi_model_stat(
                member_reduced_attr, reduction=model_reduction,
                save_as=f"{attr}_{model_reduction}_multimodel",
            )
 
        return self
    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self):
        return (
            self.load_raw()
                .concatenate_to_parents()
                .load_picontrol()
                .regrid()
                .compute_anomalies()
                .compute_global_mean()
        )

    def save(self, outdir, activity_id, array_outputs=None, scalar_outputs=None, overwrite=False):
        """
        Save selected pipeline outputs to disk under
        outdir/activity_id/<varname>/<category>/...
 
        array_outputs : dict, optional
            {category_name: (nested_dict, varname)} for Dataset/DataArray
            outputs. Defaults to concat, regridded_concat, anomaly, gsat,
            gsat_rolling under their natural variable names; any that are
            empty (not yet computed) are skipped automatically.
        scalar_outputs : dict, optional
            {category_name: nested_dict} for plain-scalar-dict outputs
            (gwl_matching, peak_warming), saved as JSON.
        overwrite : bool
            Passed through to save_processed -- see its docstring.
        """
        if array_outputs is None:
            array_outputs = {
                'concat': (self.concat, 'tas'),
                'regridded_concat': (self.regridded_concat, 'tas'),
                'anomaly': (self.anomaly, 'tas'),
                'gsat': (self.gsat, 'gsat'),
                'gsat_rolling': (self.gsat_rolling, 'gsat'),
            }
        if scalar_outputs is None:
            scalar_outputs = {
                'gwl_matching': self.gwl_matching,
                'peak_warming': self.peak_warming,
            }
 
        for category, (data, varname) in array_outputs.items():
            if not data:
                continue
            save_io.save_processed(data, outdir, activity_id, varname, category = category, overwrite=overwrite)
 
        for category, data in scalar_outputs.items():
            if not data:
                continue
            save_io.save_scalar_outputs(data, outdir, activity_id, category)