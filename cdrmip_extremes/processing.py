import numpy as np
import xarray as xr
import pandas as pd
import os
import netCDF4 as nc
import cftime
from pathlib import Path
from collections import defaultdict
import sys
import xesmf as xe
import cftime
# ---------------------------------------------------------------------------
# Concatenating
# ---------------------------------------------------------------------------

def _parse_branch_date(branch_date, calendar):
    """
    """
    if isinstance(branch_date, str):
        parts = [int(p) for p in branch_date.split('-')]
        year = parts[0]
        month = parts[1] if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else 1
        return cftime.datetime(year, month, day, calendar=calendar)
    return branch_date


def concatenate_branches(child_ds, parent_ds, time_dim='time',
                          branch_date=None, realign_years=None):
    """
    realign_years : tuple (parent_start_year, branch_year)

    """
    parent_times = parent_ds.indexes[time_dim]
    calendar = parent_times.calendar
    child_times = child_ds.indexes[time_dim]
    if calendar != child_times.calendar:
        raise ValueError(f"Calendar mismatch: parent={calendar}, child={child_times.calendar}")
    if branch_date is not None:
        branch_date = _parse_branch_date(branch_date, calendar)
    # Some ssp534-over runs are already concatenated to 2015 but still point to
    # ssp585 as their parent (e.g. CNRM-ESM2-1 and MIROC-ES2L). The below accounts for this
    # by returning the child_ds but with updated attributes
    if realign_years is None and child_times.values[0] <= parent_times[0]:
        print(f"  child already covers parent's full range from {child_times.values[0]} "
              f"(parent starts {parent_times[0]}) — skipping this hop, continuing chain "
              f"via parent's own provenance attrs")
        return child_ds.assign_attrs(
            experiment_id=parent_ds.attrs.get('experiment_id'),
            source_id=parent_ds.attrs.get('source_id'),
            parent_experiment_id=parent_ds.attrs.get('parent_experiment_id'),
            parent_variant_label=parent_ds.attrs.get('parent_variant_label'),
            parent_source_id=parent_ds.attrs.get('parent_source_id'),
            parent_time_units=parent_ds.attrs.get('parent_time_units'),
            branch_time_in_parent=parent_ds.attrs.get('branch_time_in_parent'),
        )
        
    if realign_years is not None:
        parent_start_year, branch_year = realign_years

        # shift_parent dataset
        parent_relabeled = shift_time_to_start_year(parent_ds, parent_start_year, time_dim)
        # find cutoff
        cutoff = parent_relabeled.indexes[time_dim][0].replace(year=branch_year)

        # check that parent_before has time length greater than 0 and that its final value is
        # greater than or equal to our cutoff and then index the parent to before that cutoff
        # parent_before = parent_relabeled.sel({time_dim: slice(None, cutoff)})
        parent_before = parent_relabeled.sel(
            {time_dim: parent_relabeled[time_dim] < cutoff}
        )
        if len(parent_before[time_dim]) > 0 and parent_before[time_dim].values[-1] >= cutoff:
            parent_before = parent_before.isel({time_dim: slice(None, -1)})

        if len(parent_before[time_dim]) == 0:
            raise ValueError(
                f"No parent timesteps before branch_year={branch_year} after "
                f"relabeling to start at {parent_start_year} — parent_ds is "
                f"likely too short (parent spans {parent_times[0]} to {parent_times[-1]})."
            )

        child_ds = shift_time_to_start_year(child_ds, branch_year, time_dim)

        print(f"Concatenating via realign_years={realign_years} "
              f"(parent relabeled to end {parent_before[time_dim].values[-1]}, "
              f"child relabeled to start {child_ds[time_dim].values[0]})")

    else:
        if branch_date is None:
            if 'branch_time_in_parent' not in child_ds.attrs or 'parent_time_units' not in child_ds.attrs:
                raise KeyError(
                    "child_ds is missing 'branch_time_in_parent'/'parent_time_units' "
                    "attrs and no branch_date/realign_years override was supplied."
                )
            branch_days = child_ds.attrs['branch_time_in_parent']
            units = child_ds.attrs['parent_time_units']
            branch_date = cftime.num2date(branch_days, units=units, calendar=calendar)

        child_start = child_times.values[0]
        cutoff = min(branch_date, child_start)
        parent_before = parent_ds.sel({time_dim: slice(None, cutoff)})
        if len(parent_before[time_dim]) > 0 and parent_before[time_dim].values[-1] >= child_start:
            parent_before = parent_before.isel({time_dim: slice(None, -1)})
        if len(parent_before[time_dim]) == 0:
            raise ValueError(
                f"No parent timesteps before the branch point. "
                f"branch_date={branch_date}, child_start={child_start}, "
                f"parent_ds spans {parent_times[0]} to {parent_times[-1]}."
            )

        print(f"Concatenating at {branch_date} "
              f"(parent ends {parent_before[time_dim].values[-1]}, child starts {child_start})")



    ds_concat = xr.concat([parent_before, child_ds], dim=time_dim)
    if not ds_concat.indexes[time_dim].is_monotonic_increasing:
        raise ValueError("Concatenated time axis is not monotonically increasing")

    return ds_concat


# ---------------------------------------------------------------------------
# Align Time
# ---------------------------------------------------------------------------

def shift_time_to_start_year(ds, start_year=0, time_dim='time'):
    """
    Shift ds's time coordinate by a constant number of years so the first
    timestep becomes `start_year`. Preserves month/day/hour and the
    original calendar — only the year number changes. Intended for
    aligning 1pctCO2 / 1pctCO2-cdr runs onto a common "experiment year"
    axis, since models use arbitrary native start years but are expected
    to branch at the same *elapsed* year.
    """
    times = ds[time_dim].values
    first_year = times[0].year
    offset = start_year - first_year
    calendar = times[0].calendar

    new_times = []
    for t in times:
        year, month, day = t.year + offset, t.month, t.day
        if month == 2 and day == 29 and not calendar_isleap(year, calendar):
            day = 28
        new_times.append(cftime.datetime(year, month, day, t.hour, t.minute, t.second, calendar=calendar))
    return ds.assign_coords({time_dim: new_times})

def calendar_isleap(year, calendar):
    if calendar in ('standard', 'gregorian', 'proleptic_gregorian'):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    if calendar == 'julian':
        return year % 4 == 0
    if calendar in ('noleap', '365_day'):
        return False
    if calendar in ('all_leap', '366_day'):
        return True
    if calendar == '360_day':
        return False  # no Feb 29 concept
    raise ValueError(f"unrecognized calendar: {calendar}")


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

# 180 lon, 90 lat grid
# lon_new = np.linspace(-179.5, 179.5, 180)
# lat_new = np.linspace(-89.5, 89.5, 90)
lon_new = np.linspace(0, 358, 180)
lat_new = np.linspace(-89.0, 89.0, 90)

grid_out = xr.Dataset(
    {
        "lat": (["lat"], lat_new),
        "lon": (["lon"], lon_new),
    }
)

_regridder_cache = {}

def regrid_to_common(ds, method='bilinear', periodic=True):
    # Build a cache key from the source grid shape and bounds
    cache_key = (ds.lat.shape, ds.lon.shape, 
                 float(ds.lat.min()), float(ds.lat.max()),
                 float(ds.lon.min()), float(ds.lon.max()),
                 method)
    
    if cache_key not in _regridder_cache:
        print(f"Building regridder for grid {ds.lat.shape} x {ds.lon.shape}")
        _regridder_cache[cache_key] = xe.Regridder(
            ds, grid_out, method, periodic=periodic, reuse_weights=False
        )
    
    return _regridder_cache[cache_key](ds)

def concatenate_cdr_branches(ds_down,ds_up):
    time_up = xr.cftime_range("0000-01-16",freq="1M",periods=12*140,calendar='noleap')
    len_down = len(ds_down.time)
    time_down = xr.cftime_range("0140-01-16",freq="1M",periods = len_down,calendar='noleap')

    ds_up = ds_up.isel(time=slice(None,12*140)).assign_coords({'time':time_up})
    ds_down = ds_down.assign_coords({'time':time_down})

    return xr.concat([ds_up,ds_down],dim='time')
