"""
gwl.py

GWL (global warming level) crossing-year, period-extraction, and matching
functions, extracted from CDRMIPPipeline so they can be called on any
variable's anomaly tree against a GSAT-derived reference tree that didn't
necessarily come from the same pipeline run (e.g. pr anomalies matched
against tas-derived GWL crossing years / matching years).

Every function here is a pure function of its input trees + experiment
configs -- no `self`, no assumption that the reference (gsat-derived) tree
and the target (anomaly) tree share the same variable or came from the
same CDRMIPPipeline instance.
"""

from collections import defaultdict

import xarray as xr

from . import stats, utils


def _as_dataarray(obj, varname=None):
    """
    Unwrap a Dataset back to the DataArray it was saved from.

    stats.find_crossing_years() / find_matching_gwl_year() return DataArrays
    directly when called in-memory, but a tree loaded from disk via
    load_processed() comes back as single-variable Datasets (xr.open_dataset
    always returns a Dataset). Accept either so these functions work the
    same whether the reference tree is fresh or reloaded.
    """
    if isinstance(obj, xr.Dataset):
        if varname and varname in obj:
            return obj[varname]
        data_vars = list(obj.data_vars)
        if len(data_vars) == 1:
            return obj[data_vars[0]]
        raise ValueError(
            f"Expected a DataArray or single-variable Dataset, got Dataset "
            f"with variables {data_vars}"
        )
    return obj


# ---------------------------------------------------------------------------
# GSAT-derived reference quantities
# ---------------------------------------------------------------------------

def compute_gsat_rolling(gsat_tree, experiments):
    """Rolling-mean GSAT. gsat_tree: {model: {expt: {member: ds}}} with 'gsat' in ds."""
    cfg_by_expt = {c.experiment_id: c for c in experiments}
    rolling = defaultdict(lambda: defaultdict(dict))

    for model, expt_dict in gsat_tree.items():
        for expt, member_dict in expt_dict.items():
            cfg = cfg_by_expt[expt]
            time_dim = 'time' if cfg.anomaly_time_resolution == 'native' else 'year'
            for member, ds in member_dict.items():
                if 'gsat' not in ds:
                    continue
                rolling[model][expt][member] = stats._rolling_mean(
                    ds['gsat'], window=cfg.rolling_window, time_dim=time_dim
                )

    return utils.defaultdict_to_dict(rolling)


def compute_crossing_years(gsat_tree, experiments):
    """
    GWL crossing years from GSAT. gsat_tree: {model: {expt: {member: ds}}}
    with 'gsat' in ds. Only experiments with `crossing_gwls` set are processed.
    """
    cfg_by_expt = {c.experiment_id: c for c in experiments}
    crossing_years = defaultdict(lambda: defaultdict(dict))

    for model, expt_dict in gsat_tree.items():
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

    return utils.defaultdict_to_dict(crossing_years)


def find_matching_years(gsat_tree, experiments):
    """
    GWL-matching years from GSAT: for each member, the earlier (ramp-up)
    year whose GSAT matches the level characterising `gwl_match_period`.
    gsat_tree: {model: {expt: {member: ds}}} with 'gsat' in ds. Only
    experiments with `gwl_match_period` set are processed.
    """
    cfg_by_expt = {c.experiment_id: c for c in experiments}
    matching = defaultdict(lambda: defaultdict(dict))

    for model, expt_dict in gsat_tree.items():
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
                matching[model][expt][member] = stats.find_matching_gwl_year(
                    ds['gsat'], window=cfg.rolling_window,
                    match_slice=slice(*cfg.gwl_match_period),
                    time_dim='year', search_slice=search_slice,
                )

    return utils.defaultdict_to_dict(matching)


# ---------------------------------------------------------------------------
# Applying GSAT-derived reference quantities to any variable's anomaly tree
# ---------------------------------------------------------------------------

def extract_gwl_periods(anomaly_tree, crossing_years_tree, experiments, varname='tas'):
    """
    Extract multi-year windows of `anomaly_tree[..][varname]` centred on the
    GWL crossing years in `crossing_years_tree`. The two trees need not come
    from the same run/variable -- e.g. a pr anomaly tree matched against
    tas-derived crossing years -- but must share model/expt/member keys
    (members present in one but not the other are skipped, not errored).

    Returns (gwl_periods, gwl_reversibility), both {model: {expt: {member: ...}}}.
    """
    cfg_by_expt = {c.experiment_id: c for c in experiments}
    periods = defaultdict(lambda: defaultdict(dict))
    reversibility = defaultdict(lambda: defaultdict(dict))

    for model, expt_dict in anomaly_tree.items():
        for expt, member_dict in expt_dict.items():
            cfg = cfg_by_expt[expt]
            if not cfg.crossing_gwls:
                continue
            gwl_years_by_member = crossing_years_tree.get(model, {}).get(expt, {})
            window = cfg.gwl_extract_window or cfg.rolling_window
            # matches the target variable's own time resolution, which may
            # differ from the (always-annual) resolution the crossing years
            # were computed at -- e.g. pr kept at native monthly resolution
            # while tas-derived crossing_years are annual.
            time_dim = 'time' if cfg.anomaly_time_resolution == 'native' else 'year'

            for member, ds in member_dict.items():
                if varname not in ds:
                    continue
                gwl_years = gwl_years_by_member.get(member)
                if gwl_years is None:
                    continue
                gwl_years = _as_dataarray(gwl_years)
                period_da = stats.extract_gwl_period(
                    ds[varname], gwl_years, window=window, time_dim=time_dim
                )
                periods[model][expt][member] = period_da

                if cfg.overshoot:
                    # gwl_branch_difference reduces over extract_gwl_period's own
                    # windowed axis (always 'time', independent of the source
                    # variable's time_dim) -- use its default, don't override.
                    reversibility[model][expt][member] = stats.gwl_branch_difference(period_da)

    return utils.defaultdict_to_dict(periods), utils.defaultdict_to_dict(reversibility)


def gwl_matched_differences(matching_tree, anomaly_tree, experiments, varname='tas'):
    """
    Compute the anomaly difference between `gwl_match_period` and its
    matched ramp-up year, for `anomaly_tree[..][varname]`, using
    already-known matching years in `matching_tree` (e.g. from
    find_matching_years() on a tas GSAT tree). Trees need not share a
    variable, only model/expt/member keys.

    Returns gwl_differences: {model: {expt: {member: ...}}}.
    """
    cfg_by_expt = {c.experiment_id: c for c in experiments}
    differences = defaultdict(lambda: defaultdict(dict))

    for model, expt_dict in matching_tree.items():
        for expt, member_dict in expt_dict.items():
            cfg = cfg_by_expt[expt]
            for member, matching_da in member_dict.items():
                anom_ds = anomaly_tree.get(model, {}).get(expt, {}).get(member)
                if anom_ds is None or varname not in anom_ds:
                    continue
                matching_da = _as_dataarray(matching_da)
                matching_year = int(matching_da.isel(gwl=0).sel(branch='ramp_up').values)
                time_dim = 'time' if cfg.anomaly_time_resolution == 'native' else 'year'
                differences[model][expt][member] = stats.gwl_matched_anomaly_difference(
                    anom_ds[varname], slice(*cfg.gwl_match_period), matching_year,
                    time_dim=time_dim,
                )

    return utils.defaultdict_to_dict(differences)