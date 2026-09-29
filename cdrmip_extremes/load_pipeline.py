"""
load_pipeline.py
module to expedite the loading and basic processing of ESGF data

"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Union

import xarray as xr
from .load_raw import search_and_load, search_and_load_parent
from .configs import DEFAULT_CHUNKS, DEFAULT_VARS_TO_DROP
from .processing import concatenate_branches, regrid_to_common, calc_anomaly
from . import utils, stats, io_disk, gwl

# ---------------------------------------------------------------------------
# Configuration
# Set the experimental 'configurations' specific to a given experiment (e.g. 
# 1pctCO2-cdr) to correctly load and process associated datasets
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    experiment_id: str
    table_id: str
    variable_id: str
    grid_label: str = 'gn'
    member_id: Optional[Union[str, list]] = None # if this is set to None, all members are taken
    anomaly_reference: str = 'period' # can be 'period' or 'piControl'
    reference_period: Optional[tuple] = (1850, 1900) # used only if anomaly_reference == 'period'
    anomaly_time_resolution: str = 'annual'
    stop_concat_before: tuple = ('piControl',)        # parent experiments at which to stop chaining

    rolling_window: int = 21
    
    # GWL-analysis
    crossing_gwls: Optional[list] = None        # e.g. [1.5, 2.0, 3.0]; None -> skip for this experiment
    overshoot: bool = True
    # GWL matching analysis (compute_gwl_matching) ---
    gwl_match_period: Optional[tuple] = None        # e.g. (320, 341) or (2170, 2201);
    gwl_match_search_period: Optional[tuple] = None  # None -> search pre-peak automatically

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

@dataclass
class LoadConfig: # parameters for the loading of the dataset
    experiments: list  # list the ExperimentalConfigs -> list[ExperimentConfig]
    chunks: Optional[dict] = None
    vars_to_drop: Optional[list] = None

    # Branch-date overrides (needed to account for incorrect metadata)
    branch_date_overrides: dict = field(default_factory=dict)

    realign_years_overrides: dict = field(default_factory=dict)

    # {source_id: parent_source_id}, for models whose 'parent_source_id'
    # attribute is missing/unreliable.
    source_id_overrides: Optional[dict] = None


# ---------------------------------------------------------------------------
# Loading Pipeline
# ---------------------------------------------------------------------------

class LoadingPipeline:

    def __init__(self, config: LoadConfig):
        self.config = config
        self.config.chunks = DEFAULT_CHUNKS if self.config.chunks is None else self.config.chunks
        self.config.vars_to_drop = (
            DEFAULT_VARS_TO_DROP if self.config.vars_to_drop is None else self.config.vars_to_drop
        )
        # cache to store 'parent' datasets in case the same parent is shared
        # across datasets
        self._parent_cache = {} 
        # the 'raw' (unprocessed) datasets
        self.raw = {} 
        # datasets concatenated to parents
        self.concat = {} 
        # piControl datast (needed if anomaly_reference = 'piControl')
        self.piControl = {} 
        # populated by regrid()
        self.regridded_concat = {}
        self.regridded_piControl = {}

        # populated by compute_anomalies() / compute_global_mean() / compute_gsat_rolling()
        self.anomaly = {}
        self.gsat = {}
        self.gsat_rolling = {}
        self.peak_warming = {}

        # populated by compute_crossing_years() / compute_gwl_matching() — optional,
        self.crossing_years = {}
        self.gwl_matching = {}


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
    # Step 2: concatenate to parent(s)
    # ------------------------------------------------------------------
    
    # helper to find branch_dates for datasets in which this is passed as an 
    # 'override'
    def _lookup_branch_date(self, source_id, experiment_id, parent_experiment_id):
        overrides = self.config.branch_date_overrides
        if (source_id, experiment_id) in overrides:
            return overrides[(source_id, experiment_id)]
        if parent_experiment_id in overrides:
            return overrides[parent_experiment_id]
        return None  # fall back to branch_time_in_parent metadata

    # helper to find the years (if specified) that datasets should be realigned to
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

        Note: After each concatenation, ds.attrs inherits the parent's
        attrs
        """
        while True:
            parent_experiment_id = ds.attrs.get('parent_experiment_id')
            # if there is no parent_experiment_id stop the chase
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
                    full_ds = full_ds.assign_attrs(experiment_id=cfg.experiment_id)
                    concat[model][cfg.experiment_id][member] = full_ds
        self.concat = utils.defaultdict_to_dict(concat)
        return self

    # ------------------------------------------------------------------
    # Step 3: load piControl (deduplicated across experiments per model/member)
    # ------------------------------------------------------------------
    def load_piControl(self):
        piControl = defaultdict(dict)
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
                    if member not in piControl[model]:
                        piControl[model][member] = parent_ds
                    elif not piControl[model][member].identical(parent_ds):
                        raise ValueError(f"Inconsistent piControl for {model} {member}")
        self.piControl = utils.defaultdict_to_dict(piControl)
        return self


    # ------------------------------------------------------------------
    # Step 4: regrid
    # ------------------------------------------------------------------
    def regrid(self):
        def process_ds(ds):
            return regrid_to_common(utils.fix_units(ds))
    
        self.regridded_concat = utils._map_leaves(self.concat, process_ds)
        self.regridded_piControl = utils._map_leaves(self.piControl, process_ds)
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
                varname = cfg.variable_id
                for member, ds in member_dict.items():
                    if varname not in ds:
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
                            anomaly[model][expt][member] = calc_anomaly(
                                ds_annual, ds_annual,
                                slice(*cfg.reference_period), time_dim='year'
                            )
 
                    elif cfg.anomaly_reference == 'piControl':
                        pi_ds = self.regridded_piControl.get(model, {}).get(member)
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
        cfg_by_expt = {c.experiment_id: c for c in self.config.experiments}
        gsat = defaultdict(lambda: defaultdict(dict))
    
        for model, expt_dict in self.anomaly.items():
            for expt, member_dict in expt_dict.items():
                varname = cfg_by_expt[expt].variable_id
                for member, ds in member_dict.items():
                    if varname not in ds:
                        gsat[model][expt][member] = ds
                        continue
                    gsat[model][expt][member] = utils.global_mean(ds).to_dataset(name='gsat')
    
        self.gsat = utils.defaultdict_to_dict(gsat)
        return self
    # ------------------------------------------------------------------
    # Optional: force computation of gsat
    # ------------------------------------------------------------------
    def load_gsat(self):
        self.gsat = utils._map_leaves(self.gsat, lambda ds: ds.load())
        return self
    
    def load(self, attribute):
        setattr(self, attribute, utils.map_nested(lambda x: x.load(), getattr(self, attribute)))
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
 
        self.peak_warming = utils._map_leaves(self.gsat_rolling, peak_fn)
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
                    crossing_years[model][expt][member] = gwl.find_crossing_years(
                        ds['gsat'], window=cfg.rolling_window,
                        gwls=cfg.crossing_gwls, time_dim='year',
                        overshoot=cfg.overshoot,
                    )
 
        self.crossing_years = utils.defaultdict_to_dict(crossing_years)
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
                    matching_da = gwl.find_matching_gwl_year(
                        ds['gsat'], window=cfg.rolling_window,
                        match_slice=slice(*cfg.gwl_match_period),
                        time_dim='year', search_slice=search_slice,
                    )
                    matching[model][expt][member] = matching_da

        self.gwl_matching = utils.defaultdict_to_dict(matching)
        return self
    # ------------------------------------------------------------------
    # Run it all in one step
    # ------------------------------------------------------------------
    def run(self):
        return (
            self.load_raw()
                .concatenate_to_parents()
                .load_piControl()
                .regrid()
                .compute_anomalies()
                .compute_global_mean()
        )
    # ------------------------------------------------------------------
    # Optional - Save
    # ------------------------------------------------------------------
    def save_tree(self, attribute, varname, proc_dir, activity_id, category=None, overwrite=False):
        io_disk.save_tree(
            getattr(self, attribute),
            proc_dir,
            varname,
            activity_id=activity_id,
            category=category,
            overwrite=overwrite,
        )