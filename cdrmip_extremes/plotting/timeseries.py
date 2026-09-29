"""
timeseries.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from .core import as_dataarray, maybe_save

__all__ = ["plot_gsat_timeseries"]

MARKER_KW = dict(s=400, zorder=5)


def _members(node):
    """List of per-member leaves; a single leaf if the tree is already member-reduced."""
    if isinstance(node, (xr.DataArray, xr.Dataset)):
        return [node]
    if getattr(node, "has_data", False):  # DataTree leaf
        return [node.ds]
    return list(node.values())


def _median_gwl_matching(node):
    """(median gwl value, median ramp-up matching year) across members."""
    das = [as_dataarray(v, "gwl_matching_years") for v in _members(node)]
    gwl_val = np.round(np.median([float(da.gwl) for da in das]), 2)
    years = xr.concat([da.drop_vars("gwl") for da in das], dim="member").median("member")
    return gwl_val, float(years.sel(branch="ramp_up"))


def _median_crossing_years(node):
    """crossing_years (dims gwl, branch) as the median across members."""
    das = [as_dataarray(v, "crossing_years") for v in _members(node)]
    return xr.concat(das, dim="member").median("member")


def _draw_gwl_markers(ax, mode, plotted, gwl_matching_years, colour_dict, end_marker_year,
                      crossing_years, gwl):
    if mode not in ("height", "vline"):
        raise ValueError(f"Unknown gwl_marker_mode {mode!r}")

    if mode == "vline" and crossing_years is not None and gwl is not None:
        # ramp-up (^) and ramp-down (v) crossings of `gwl`, pinned to the curve
        for model, expt_dict in crossing_years.items():
            if model not in plotted:
                continue
            for node in expt_dict.values():
                reduced = _median_crossing_years(node)
                for branch, marker in (("ramp_up", "^"), ("ramp_down", "v")):
                    year = float(reduced.sel(branch=branch, gwl=gwl))
                    if not np.isnan(year):
                        ax.scatter(year, float(plotted[model].interp(year=year)),
                                   color=colour_dict[model], marker=marker, **MARKER_KW)
        return

    for model, expt_dict in (gwl_matching_years or {}).items():
        for node in expt_dict.values():
            gwl_val, up_year = _median_gwl_matching(node)
            colour = colour_dict[model]
            if mode == "height":
                # markers sit at y = gwl; only meaningful when the y-axis is GSAT
                ax.scatter(end_marker_year, gwl_val, color=colour, marker="v", **MARKER_KW)
                ax.scatter(up_year, gwl_val, color=colour, marker="^", **MARKER_KW)
            elif model in plotted:
                # fallback: ramp-up crossing only, pinned to the curve
                ax.scatter(up_year, float(plotted[model].interp(year=up_year)),
                           color=colour, marker="^", **MARKER_KW)


def plot_gsat_timeseries(gsat_rolling, colour_dict,
                         varname="gsat", title=None,
                         ylabel="GSAT (°C)", xlabel="Time", xlim=(1850, 2300), ylim=(-0.2, 4.1),
                         xticks=None, end_marker_year=2190, figsize=(16, 8), legend_ncol=5,
                         gwl_marker_mode="height", crossing_years=None, gwl_matching_years=None, gwl=None,
                         raw_tree=None, raw_alpha=0.3,
                         co2=None, co2_label="Atmospheric CO$_2$ (ppm)",
                         legend_loc="below", legend_fontsize=None,
                         fig=None, ax=None, fig_dir=None, folder=None, file_name=None):
    """
    Rolling timeseries with GWL crossing markers, for any tree shaped
    {model: {expt: Dataset-or-DataArray}} (`varname` picks the variable).

    gwl_marker_mode:
      'height' -- markers at y=gwl from `gwl_matching_years` (ramp-up year and
                  `end_marker_year`). Only sensible when the y-axis is GSAT.
      'vline'  -- markers pinned to the curve itself. With `crossing_years` and
                  `gwl`, draws both ramp-up (^) and ramp-down (v) crossings;
                  otherwise only the ramp-up year from `gwl_matching_years`.
    `crossing_years` / `gwl_matching_years` may be per-member
    ({model: {expt: {member: ds}}}, median taken here) or already member-reduced.

    `raw_tree`: unsmoothed tree of the same shape, drawn faded underneath.
    `co2`: DataArray drawn on a twin y-axis against its 'year' coord.
    `legend_loc`: 'below', any matplotlib loc, or None for no legend.
    Pass `fig`/`ax` to draw into an existing axes (e.g. a composite figure).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if raw_tree is not None:
        for model, expt_dt in raw_tree.items():
            for node in expt_dt.values():
                as_dataarray(node, varname=varname).plot(
                    ax=ax, color=colour_dict[model], linewidth=2, alpha=raw_alpha)

    plotted = {}
    for model, expt_dt in gsat_rolling.items():
        for node in expt_dt.values():
            plotted[model] = as_dataarray(node, varname=varname)
            plotted[model].plot(ax=ax, label=model, linewidth=3, color=colour_dict[model])

    _draw_gwl_markers(ax, gwl_marker_mode, plotted, gwl_matching_years, colour_dict,
                      end_marker_year, crossing_years, gwl)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    for x in (140, 280):
        ax.axvline(x=x, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=1.0)

    handles, labels = ax.get_legend_handles_labels()
    if co2 is not None:
        ax2 = ax.twinx()
        co2.plot(ax=ax2, color="gray", linewidth=2, linestyle="--", alpha=0.7,
                 label="Atmospheric CO$_2$")
        ax2.set_ylabel(co2_label, fontsize=18, color="gray")
        ax2.set_xlabel("")
        ax2.set_title("")
        ax2.tick_params(axis="y", colors="gray", labelsize=16)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    unique = dict(zip(labels, handles))
    if legend_loc == "below":
        ax.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=legend_ncol, fontsize=legend_fontsize or 18, frameon=False)
    elif legend_loc is not None:
        ax.legend(unique.values(), unique.keys(), loc=legend_loc, ncol=1,
                  fontsize=legend_fontsize or 12, frameon=False)
    ax.set_title(title, fontsize=24, y=1.02)

    if standalone:
        fig.tight_layout()
        maybe_save(fig, fig_dir, folder, file_name)
    return fig