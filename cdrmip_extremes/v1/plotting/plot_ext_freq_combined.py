"""
plot_ext_freq_combined.py

A single figure that stacks:
  * top:    plotting.plot_ext_freq()      -> the 3x3 (threshold x branch/difference)
                                             grid for one model, panels (a)(b)(c)
  * bottom: plotting.plot_ext_freq_diff()  -> the per-model difference grid for one
                                             threshold, labelled as a single panel (d)

into one matplotlib Figure using SubFigures.

Drop this function into cdrmip_extremes/plotting.py (it reuses the module's own
helpers: get_ds, make_norm, plot_map_panel, add_stippling, add_colorbar,
hide_unused_axes, save_figure and the module constants EXT_TYPE_PARAMS / THRESHOLDS).
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs


def plot_ext_freq_with_model_diffs(
    exceedances, differences, agreement,
    model, expt, ext_type,
    diff_threshold, diff_vmin, diff_vmax,
    gwl=None, period=None,
    model_name=None, discrete=False, levels=None, levels_diff=None,
    n_cols=4,
    exclude_models=("multi_model_median", "Multi-Model Median"),
    diff_label="(d)",
    diff_title=None,
    projection=None,
    suptitle=None, expt_name=None,
    fig_dir=None, folder=None, file_name=None,
):
    """Combine plot_ext_freq (top) and plot_ext_freq_diff (bottom) in one figure.

    Top block (== plot_ext_freq):
        A 3x3 grid for `model`: rows = sigma thresholds, columns =
        ramp up / ramp down / difference, with agreement stippling on the
        difference column. Panels labelled (a) (b) (c).

    Bottom block (== plot_ext_freq_diff):
        One difference map per model (ramp_down - ramp_up) at a single
        `diff_threshold`, in an n_cols-wide grid. The whole block is
        labelled as one panel, `diff_label` (default "(d)").

    Parameters mirror the two originals. The ones specific to the bottom
    block are prefixed diff_:
        diff_threshold : which sigma threshold the per-model grid shows
                         (e.g. 'sigma1')
        diff_vmin/diff_vmax : colour limits for the per-model difference maps
                              (the originals pass these as vmin/vmax)
        diff_label : corner label for the whole bottom block ("(d)")
        diff_title : short symbol shown above the block (defaults to the
                     threshold symbol from EXT_TYPE_PARAMS)

    gwl / period : exactly one must be given, same semantics as the originals
        (gwl -> .sel(gwl=gwl); period -> .isel(gwl=0)).

    fig_dir/folder/file_name : if all three given, the combined figure is saved.
    """
    # ---- validation / setup (shared) -------------------------------------
    if gwl is None and period is None:
        raise ValueError("Provide either gwl (crossing_years-based) or period (gwl_matching-based).")
    if model_name is None:
        model_name = model
    if expt_name is None:
        expt_name = expt
    projection = projection or ccrs.Robinson()

    def _select(da):
        return da.isel(gwl=0) if period is not None else da.sel(gwl=gwl)

    params = EXT_TYPE_PARAMS[ext_type]

    # ---- how many rows will the bottom (per-model) block need? -----------
    diff_models = [m for m in differences.keys() if m not in exclude_models]
    n_plots_bottom = len(diff_models)
    n_rows_bottom = max(1, math.ceil(n_plots_bottom / n_cols))

    # ---- parent figure split into two stacked subfigures -----------------
    top_h = 10.0                      # matches plot_ext_freq's figsize height
    bottom_h = n_rows_bottom * 3.0    # ~ plot_ext_freq_diff's n_rows * 2.85 + slack
    fig = plt.figure(figsize=(16, top_h + bottom_h + 1.0))
    sf_top, sf_bottom = fig.subfigures(
        2, 1, height_ratios=[top_h, bottom_h + 1.0]
    )

    # =====================================================================
    # TOP BLOCK  ==  plot_ext_freq()
    # =====================================================================
    exceedance = get_ds(exceedances, model, expt, ext_type)
    diff = get_ds(differences, model, expt, ext_type)
    agreement_ds = get_ds(agreement, expt, ext_type)

    to_plot = {}
    for threshold in THRESHOLDS:
        to_plot[threshold] = {
            "ramp_up": _select(exceedance[threshold]).sel(branch="ramp_up"),
            "ramp_down": _select(exceedance[threshold]).sel(branch="ramp_down"),
            "difference": _select(diff[threshold]),
        }

    levels1 = levels if levels is not None else params["levels"]
    levels2 = levels_diff if levels_diff is not None else params["levels_diff"]

    if discrete:
        cmap1, norm1 = make_norm(levels1, params["cmap1"])
        cmap2, norm2 = make_norm(levels2, params["cmap2"])
        cmap2.set_extremes(under=params["diff_under"], over=params["diff_over"])
    else:
        cmap1 = plt.get_cmap(params["cmap1"])
        cmap2 = plt.get_cmap(params["cmap2"])
        norm1 = Normalize(vmin=levels1[0], vmax=levels1[-1])
        norm2 = Normalize(vmin=levels2[0], vmax=levels2[-1])

    axes_top = sf_top.subplots(
        3, 3, subplot_kw={"projection": projection}, squeeze=False, sharey=True,
    )
    # leave room at the bottom of the top subfigure for its two colorbars
    sf_top.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.14,
                           wspace=0.08, hspace=0.12)

    for row_index, threshold in enumerate(THRESHOLDS):
        for col_index, key in enumerate(["ramp_up", "ramp_down", "difference"]):
            ax = axes_top[row_index, col_index]
            norm, cmap = (norm2, cmap2) if key == "difference" else (norm1, cmap1)
            plot_map_panel(ax, to_plot[threshold][key], cmap=cmap, norm=norm,
                           method="imshow", wrap_cyclic=False)
        axes_top[row_index, 0].annotate(
            params["threshold_symbs"][row_index], xy=(-0.025, 0.5),
            xycoords="axes fraction", ha="right", va="center",
            fontsize=20, rotation=0,
        )

    if expt == "ssp534-over":
        axes_top[0][0].set_title("pre-overshoot", fontsize=20)
        axes_top[0][1].set_title("post-overshoot", fontsize=20)
        axes_top[0][2].set_title("difference (post-pre)", fontsize=20)
    elif expt == "1pctCO2-cdr":
        axes_top[0][0].set_title("Ramp Up", fontsize=20)
        axes_top[0][1].set_title("Ramp Down", fontsize=20)
        axes_top[0][2].set_title("Ramp Down - Ramp Up", fontsize=20)

    for row_index, threshold in enumerate(THRESHOLDS):
        add_stippling(axes_top[row_index][2], _select(agreement_ds[threshold]))

    # panel letters (a)(b)(c) on the top row
    for ax, lab in zip(axes_top[0], ["(a)", "(b)", "(c)"]):
        ax.text(0.0, 1.10, lab, transform=ax.transAxes, fontsize=20,
                va="bottom", ha="left")

    # two horizontal colorbars along the bottom of the TOP subfigure
    add_colorbar(
        sf_top, plt.cm.ScalarMappable(norm=norm1, cmap=cmap1),
        rect=[0.06, 0.06, 0.6, 0.022],
        label="Frequency of occurrence (% of years within window)",
    )
    add_colorbar(
        sf_top, plt.cm.ScalarMappable(norm=norm2, cmap=cmap2),
        rect=[0.70, 0.06, 0.265, 0.022],
        label="Change in frequency (difference in % of years)",
        extend="both",
        ticks=params.get("diff_ticks"),
        ticklabels=params.get("diff_ticklabels"),
    )

    # =====================================================================
    # BOTTOM BLOCK  ==  plot_ext_freq_diff()  (labelled as one panel: (d))
    # =====================================================================
    if ext_type == "heat":
        cmap_d = plt.cm.RdBu_r.copy()
        cmap_d.set_extremes(under="midnightblue", over="maroon")
    elif ext_type == "cold":
        cmap_d = plt.cm.RdBu.copy()
        cmap_d.set_extremes(under="maroon", over="midnightblue")
    else:
        raise ValueError(f"Unknown ext_type: {ext_type}")

    to_plot_bottom = {
        m: _select(get_ds(differences, m, expt, ext_type)[diff_threshold])
        for m in diff_models
    }

    axes_bottom = sf_bottom.subplots(
        n_rows_bottom, n_cols, subplot_kw={"projection": projection},
        squeeze=False, sharey=True,
    )
    sf_bottom.subplots_adjust(left=0.02, right=0.93, top=0.88, bottom=0.05,
                              wspace=0.08, hspace=0.15)

    mesh = None
    for (m, da), ax in zip(to_plot_bottom.items(), axes_bottom.flat):
        mesh = plot_map_panel(
            ax, da, cmap_d, vmin=diff_vmin, vmax=diff_vmax,
            title=m, title_fontsize=16, method="imshow",
        )
    hide_unused_axes(axes_bottom, n_plots_bottom)

    if mesh is not None:
        add_colorbar(
            sf_bottom, mesh,
            rect=[0.945, 0.08, 0.011, 0.8],
            label="Difference in % of years",
            orientation="vertical", extend="both",
            fraction=0.02, aspect=20, tick_fontsize=14,
        )

    # single corner label for the whole bottom block
    if diff_title is None:
        diff_title = params["threshold_symbs"][THRESHOLDS.index(diff_threshold)]
    sf_bottom.text(0.0, 0.95, diff_label, fontsize=20, va="bottom", ha="left")
    sf_bottom.text(0.5, 0.95,
                   f"{diff_title}-level {ext_type} events, per model",
                   fontsize=18, va="bottom", ha="center")

    # ---- overall title & save --------------------------------------------
    if suptitle:
        fig.suptitle(suptitle, y=0.995, fontsize=25)
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    return fig