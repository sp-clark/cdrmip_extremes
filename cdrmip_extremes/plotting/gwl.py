"""
gwl.py 
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from .core import (
    MMM, as_dataarray, select_gwl, period_phrase, display_name, mmm_first, make_norm,
    make_geo_grid, plot_map_panel, add_stippling, add_colorbar, hide_unused_axes, maybe_save,
)
from .timeseries import plot_gsat_timeseries

__all__ = ["plot_gwl_diff", "plot_gwl_diff_with_timeseries"]


def _prep(node, varname=None, gwl=None, period=None):
    """2-D lat/lon DataArray, selecting the warming level if there is a gwl dimension."""
    da = as_dataarray(node, varname=varname)
    return select_gwl(da, gwl, period) if "gwl" in da.dims else da


def plot_gwl_diff(data_dict, expt, varname=None, gwl=None, period=None, n_cols=3, levels=None,
                  cmap=None, cbar_label="SAT difference (°C)", suptitle=None, agreement=None,
                  figsize=None,titles=None,
                 fig_dir=None, folder=None, file_name=None):
    """
    One difference map per model, multi-model median first and stippled with
    `agreement`. `data_dict` is {model: {expt: Dataset-or-DataArray}}.
    """
    to_plot = {m: _prep(node[expt], varname, gwl, period)
               for m, node in data_dict.items() if expt in node}
    models = mmm_first(to_plot)
    if agreement is not None:
        agreement = _prep(agreement, varname, gwl, period)

    levels = np.linspace(-1, 1, 21) if levels is None else levels
    cmap, norm = make_norm(levels, cmap or cm.RdBu_r)
    fig, axes = make_geo_grid(math.ceil(len(models) / n_cols), n_cols,
                              figsize=figsize, layout="constrained")

    for model, ax in zip(models, axes.flat):
        title = (titles or {}).get(model, display_name(model))
        mesh = plot_map_panel(ax, to_plot[model], cmap, norm, title=title,
                              title_fontsize=20, method="imshow")
        if agreement is not None and model == MMM:
            add_stippling(ax, agreement)
    hide_unused_axes(axes, len(models))

    add_colorbar(fig, mesh, ax=axes, label=cbar_label, extend="both",
                 location="bottom", shrink=1, aspect=40, pad=0.04,
                 fontsize=18, tick_fontsize=16)
    fig.suptitle(suptitle or f"{expt} {varname} differences {period_phrase(period, gwl)}",
                 fontsize=24, y=1.05)
    maybe_save(fig, fig_dir, folder, file_name)
    return fig, axes


def plot_gwl_diff_with_timeseries(
    ts_tree, diffs, colour_dict, varname=None, diff_var=None, agreement=None,
    gwl=None, period="gwl", models=None,
    crossing_years=None, gwl_matching_years=None, gwl_marker_mode=None, end_marker_year=330,
    raw_tree=None, co2=None, co2_label="Atmospheric CO$_2$ (ppm)",
    ts_ylabel="GSAT Anomaly (°C)", ts_xlim=(0, 345), ts_ylim=None,
    xticks=(0, 140, 280, 340), legend_loc=None, legend_fontsize=None,
    levels=None, cmap=None, cbar_label="SAT Difference (°C)", n_cols=4, figsize=(24, 14),
    fig_dir=None, folder=None, file_name=None,
):
    """
    (a) timeseries of `ts_tree` with GWL markers, (b) multi-model median
    difference map with agreement stippling, (c) one difference map per model.

    ts_tree / raw_tree : {model: {expt: node}} rolling / unsmoothed global means
                         (`varname` picks the variable).
    diffs              : {model: node} difference fields including 'multi_model_median'
                         (`diff_var` picks the variable, also used for `agreement`).
    period='gwl'       : select `gwl`; markers at the ramp-up/ramp-down crossings
                         from `crossing_years`, pinned to the curve.
    period='final'     : end-of-simulation differences; markers from
                         `gwl_matching_years` (ramp-up year and `end_marker_year`).
    gwl_marker_mode overrides that choice, e.g. 'vline' with period='final' for a
    non-GSAT timeseries.
    """
    maps = {m: _prep(node, diff_var, gwl, period) for m, node in diffs.items()}
    models = models or [m for m in maps if m != MMM]
    mode = gwl_marker_mode or ("vline" if period == "gwl" else "height")

    levels = np.linspace(-2, 2, 21) if levels is None else levels
    cmap, norm = make_norm(levels, cmap or cm.RdBu_r)

    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.8], hspace=0.15)
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0], wspace=0.2,
                                              width_ratios=[1.45, 1])
    ax_ts = fig.add_subplot(top_gs[0, 0])
    ax_mmm = fig.add_subplot(top_gs[0, 1], projection=ccrs.Robinson())
    n_cols = min(n_cols, len(models)) or 1
    bottom_gs = gridspec.GridSpecFromSubplotSpec(math.ceil(len(models) / n_cols), n_cols,
                                                 subplot_spec=outer_gs[1], hspace=0.05, wspace=0.05)

    # (a) timeseries
    plot_gsat_timeseries(
        ts_tree, colour_dict, varname=varname,
        ylabel=ts_ylabel, xlabel="Year", xlim=ts_xlim, ylim=ts_ylim, xticks=xticks,
        end_marker_year=end_marker_year, gwl_marker_mode=mode,
        crossing_years=crossing_years,gwl_matching_years=gwl_matching_years, gwl=gwl, raw_tree=raw_tree,
        co2=co2, co2_label=co2_label, legend_loc=legend_loc, legend_fontsize=legend_fontsize,
        fig=fig, ax=ax_ts,
    )

    # (b) multi-model median
    im = plot_map_panel(ax_mmm, maps[MMM], cmap, norm, title=display_name(MMM),
                        title_fontsize=20, method="imshow")
    if agreement is not None:
        add_stippling(ax_mmm, _prep(agreement, diff_var, gwl, period))

    # (c) per-model maps
    for i, model in enumerate(models):
        ax = fig.add_subplot(bottom_gs[i], projection=ccrs.Robinson())
        plot_map_panel(ax, maps[model], cmap, norm, title=model, title_fontsize=18,
                       method="imshow")

    add_colorbar(fig, im, rect=[0.93, 0.1, 0.02, 0.8], label=cbar_label, orientation="vertical",
                 extend="both", fontsize=18, tick_fontsize=16)
    for text, xy in [("(a)", (-0.1, 1.02)), ("(b)", (1.1, 1.02)), ("(c)", (-0.1, -0.2))]:
        ax_ts.text(*xy, text, transform=ax_ts.transAxes, fontsize=20, va="bottom", ha="left")

    fig.tight_layout()
    maybe_save(fig, fig_dir, folder, file_name)
    return fig