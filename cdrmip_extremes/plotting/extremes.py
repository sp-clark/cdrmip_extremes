"""
extremes.py -- extreme-event frequency maps, hottest/coldest month and
preindustrial extreme statistics.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, Normalize
import cartopy.crs as ccrs

from .core import (
    MMM, mmm_first, get_ds, average_over_dim, average_leaves, select_gwl, period_phrase,
    display_name, make_norm, make_geo_grid, plot_map_panel, add_stippling,
    add_colorbar, label_panels, hide_unused_axes, maybe_save,
)

__all__ = [
    "EXT_TYPE_PARAMS", "THRESHOLDS", "MONTH_LABELS", "threshold_symb", "diff_cmap",
    "plot_ext_freq", "plot_ext_freq_diff", "plot_ext_freq_diff_corr_all_thresholds",
    "plot_ext_freq_diff_stacked", "plot_ext_freq_with_model_diffs",
    "plot_hottest_coldest_month", "plot_pi_stats",
]

EXT_TYPE_PARAMS = {
    "heat": dict(
        threshold_symbs=["1$\\sigma$", "2$\\sigma$", "3$\\sigma$"],
        levels=np.linspace(0, 100, 11),
        levels_diff=np.linspace(-50, 50, 21),
        cmap1="YlOrRd",
        cmap2="RdBu_r",
        diff_under="midnightblue",
        diff_over="maroon",
    ),
    "cold": dict(
        threshold_symbs=["-0.25$\\sigma$", "-0.5$\\sigma$", "-1$\\sigma$"],
        levels=np.linspace(0, 60, 25),
        levels_diff=np.linspace(-30, 30, 25),
        cmap1="YlGnBu",
        cmap2="RdBu",
        diff_under="maroon",
        diff_over="midnightblue",
        diff_ticks=[-30, -20, -10, 0, 10, 20, 30],
        diff_ticklabels=["-30%", "-20%", "-10%", "0", "10%", "20%", "30%"],
    ),
}

THRESHOLDS = ["sigma1", "sigma2", "sigma3"]

COLUMN_TITLES = {
    "ssp534-over": ["pre-overshoot", "post-overshoot", "difference (post-pre)"],
    "1pctCO2-cdr": ["Ramp Up", "Ramp Down", "Ramp Down - Ramp Up"],
}

EXCLUDE = (MMM, "Multi-Model Median")
DIFF_LABEL = "Difference in % of years"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def threshold_symb(ext_type, threshold):
    return EXT_TYPE_PARAMS[ext_type]["threshold_symbs"][THRESHOLDS.index(threshold)]


def diff_cmap(ext_type):
    """Diverging cmap for frequency differences, with under/over colours set."""
    p = EXT_TYPE_PARAMS[ext_type]
    cmap = plt.get_cmap(p["cmap2"]).copy()
    cmap.set_extremes(under=p["diff_under"], over=p["diff_over"])
    return cmap


def _ext_freq_styles(ext_type, discrete, levels, levels_diff):
    """(cmap, norm) for the frequency columns and for the difference column."""
    p = EXT_TYPE_PARAMS[ext_type]
    levels = p["levels"] if levels is None else levels
    levels_diff = p["levels_diff"] if levels_diff is None else levels_diff
    if discrete:
        return make_norm(levels, p["cmap1"]), make_norm(levels_diff, diff_cmap(ext_type))
    return ((plt.get_cmap(p["cmap1"]), Normalize(levels[0], levels[-1])),
            (plt.get_cmap(p["cmap2"]), Normalize(levels_diff[0], levels_diff[-1])))


def _draw_ext_freq_grid(axes, exceedances, differences, agreement, model, expt, ext_type,
                        gwl, period, styles):
    """Fill a 3x3 axes grid: rows = thresholds, cols = ramp up / ramp down / difference."""
    exceedance = get_ds(exceedances, model, expt, ext_type)
    diff = get_ds(differences, model, expt, ext_type)
    agree = get_ds(agreement, expt, ext_type)
    (cmap1, norm1), (cmap2, norm2) = styles

    for row, threshold in zip(axes, THRESHOLDS):
        exc = select_gwl(exceedance[threshold], gwl, period)
        panels = [(exc.sel(branch="ramp_up"), cmap1, norm1),
                  (exc.sel(branch="ramp_down"), cmap1, norm1),
                  (select_gwl(diff[threshold], gwl, period), cmap2, norm2)]
        for ax, (da, cmap, norm) in zip(row, panels):
            plot_map_panel(ax, da, cmap, norm=norm, method="imshow")
        add_stippling(row[2], select_gwl(agree[threshold], gwl, period))

    for ax, title in zip(axes[0], COLUMN_TITLES.get(expt, [])):
        ax.set_title(title, fontsize=20)


def _add_ext_freq_colorbars(fig, styles, ext_type, rects, labels):
    p = EXT_TYPE_PARAMS[ext_type]
    (cmap1, norm1), (cmap2, norm2) = styles
    add_colorbar(fig, cm.ScalarMappable(norm=norm1, cmap=cmap1), rect=rects[0], label=labels[0])
    add_colorbar(fig, cm.ScalarMappable(norm=norm2, cmap=cmap2), rect=rects[1], label=labels[1],
                 extend="both", ticks=p.get("diff_ticks"), ticklabels=p.get("diff_ticklabels"))


def _draw_model_diffs(axes, differences, models, expt, ext_type, threshold, gwl, period,
                      vmin, vmax, title=display_name, title_fontsize=16):
    """One difference map per model; returns the last mesh for the colorbar."""
    cmap = diff_cmap(ext_type)
    mesh = None
    for model, ax in zip(models, axes.flat):
        da = select_gwl(get_ds(differences, model, expt, ext_type)[threshold], gwl, period)
        mesh = plot_map_panel(ax, da, cmap, vmin=vmin, vmax=vmax, title=title(model),
                              title_fontsize=title_fontsize, method="imshow")
    hide_unused_axes(axes, len(models))
    return mesh


def _corr_title(model, ext_type, threshold, corr_dict):
    corr = (corr_dict or {}).get(model, {}).get(ext_type)
    if corr is None:
        return display_name(model)
    if hasattr(corr, "dims") and "threshold" in corr.dims:
        corr = corr.sel(threshold=threshold)
    elif np.ndim(corr) > 0:
        corr = corr[THRESHOLDS.index(threshold)]
    return f"{display_name(model)}, corr={float(corr):.2f}"


# ---------------------------------------------------------------------------
# Recipes: extreme frequency
# ---------------------------------------------------------------------------

def plot_ext_freq(exceedances, differences, agreement, model, expt, ext_type, gwl=None, period=None,
                  model_name=None, discrete=False, levels=None, levels_diff=None,
                  wrap_cyclic=None, fig_dir=None, folder=None, file_name=None,
                  expt_name=None, suptitle=None):
    """
    3x3 grid: rows = sigma thresholds, cols = ramp up / ramp down / difference (stippled).
    `wrap_cyclic` is unused (kept so old calls still work).
    """
    styles = _ext_freq_styles(ext_type, discrete, levels, levels_diff)
    fig, axes = make_geo_grid(3, 3, figsize=(16, 10))
    _draw_ext_freq_grid(axes, exceedances, differences, agreement, model, expt, ext_type,
                        gwl, period, styles)

    for ax, symb in zip(axes[:, 0], EXT_TYPE_PARAMS[ext_type]["threshold_symbs"]):
        ax.annotate(symb, xy=(-0.025, 0.5), xycoords="axes fraction",
                    ha="right", va="center", fontsize=20)
    label_panels(axes)

    _add_ext_freq_colorbars(
        fig, styles, ext_type,
        rects=([0.057, -0.02, 0.6, 0.025], [0.7, -0.02, 0.265, 0.025]),
        labels=("Frequency of occurrence (% of years within window)",
                "Change in frequency (difference in % of years)"),
    )

    if suptitle is None:
        suffix = {"final": "\nby End of Simulation", "equivalent": ""}.get(period, f"\nat {gwl}°C")
        suptitle = (f"{expt_name or expt} {model_name or model} Change in "
                    f"{ext_type.capitalize()} Extreme Frequency{suffix}")
    fig.suptitle(suptitle, y=1.01, fontsize=25)
    fig.tight_layout()
    maybe_save(fig, fig_dir, folder, file_name)
    return fig


def plot_ext_freq_diff(differences, expt, ext_type, threshold, levels, label, title, vmin, vmax,
                       gwl=None, period=None, exclude_models=EXCLUDE, n_cols=4,
                       fig_dir=None, folder=None, file_name=None):
    """
    Per-model difference maps (e.g. ramp_down - ramp_up) for one threshold.
    `levels` is unused (kept so old positional calls still work).
    """
    models = [m for m in differences if m not in exclude_models]
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(17, n_rows * 2.85))

    mesh = _draw_model_diffs(axes, differences, models, expt, ext_type, threshold,
                             gwl, period, vmin, vmax)
    add_colorbar(fig, mesh, rect=[1.01, 0.06, 0.01, 0.8], label=DIFF_LABEL,
                 orientation="vertical", extend="both")

    fig.text(0, 0.95, label, fontsize=20, va="bottom", ha="left")
    fig.suptitle(f"{expt} change in frequency of {title}-level {ext_type} events "
                 f"{period_phrase(period, gwl)}", fontsize=24, y=1.0)
    fig.tight_layout()
    maybe_save(fig, fig_dir, folder, file_name)
    return fig


def plot_ext_freq_diff_corr_all_thresholds(differences, expt, ext_type, levels, title, vmin, vmax,
                                           gwl=None, period=None, exclude_models=(),
                                           agreement=None, suptitle=None,
                                           n_cols=3, fig_dir=None, folder=None, file_name=None,
                                           corr_dict=None):
    """
    Per-model difference maps for all three thresholds, one block of rows per threshold,
    multi-model median first (stippled with `agreement` if given).
    `levels` and `title` are unused (kept so old positional calls still work).
    """
    models = mmm_first(m for m in differences if m not in exclude_models)
    rows_per_threshold = math.ceil(len(models) / n_cols)
    n_rows = rows_per_threshold * len(THRESHOLDS)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(n_cols * 5.7, n_rows * 2.85))

    for i, threshold in enumerate(THRESHOLDS):
        block = axes[i * rows_per_threshold:(i + 1) * rows_per_threshold, :]
        mesh = _draw_model_diffs(
            block, differences, models, expt, ext_type, threshold, gwl, period, vmin, vmax,
            title=lambda m: _corr_title(m, ext_type, threshold, corr_dict), title_fontsize=20,
        )
        if agreement is not None and models[0] == MMM:
            add_stippling(block[0, 0], select_gwl(get_ds(agreement, expt, ext_type)[threshold], gwl, period))
        block[0, 0].annotate(threshold_symb(ext_type, threshold), xy=(-0.05, -0.1),
                             xycoords="axes fraction", ha="right", va="center", fontsize=30)

    add_colorbar(fig, mesh, rect=[1.0, 0.0, 0.02, 0.95], label=DIFF_LABEL,
                 orientation="vertical", extend="both")
    fig.suptitle(suptitle or f"{expt} change in frequency of extreme {ext_type} events "
                             f"{period_phrase(period, gwl)}", fontsize=24, y=1.0)
    fig.tight_layout()
    maybe_save(fig, fig_dir, folder, file_name)
    return fig


def plot_ext_freq_diff_stacked(differences, expt, ext_type, panels, vmin, vmax,
                               gwl=None, period=None, exclude_models=EXCLUDE, n_cols=4,
                               fig_dir=None, folder=None, file_name=None):
    """
    Several plot_ext_freq_diff grids stacked vertically, one per entry in
    `panels` (dicts with keys 'threshold', 'label', 'title').
    """
    models = [m for m in differences if m not in exclude_models]
    n_rows = math.ceil(len(models) / n_cols)
    fig = plt.figure(figsize=(17, len(panels) * (n_rows * 2.45 + 1.0)))
    subfigs = np.atleast_1d(fig.subfigures(len(panels), 1, hspace=0.03))

    for subfig, panel in zip(subfigs, panels):
        axes = subfig.subplots(n_rows, n_cols, subplot_kw={"projection": ccrs.Robinson()},
                               squeeze=False)
        mesh = _draw_model_diffs(axes, differences, models, expt, ext_type, panel["threshold"],
                                 gwl, period, vmin, vmax)
        add_colorbar(subfig, mesh, rect=[0.93, 0.10, 0.01, 0.70], label=DIFF_LABEL,
                     orientation="vertical", extend="both")
        subfig.text(0.005, 0.97, panel["label"], fontsize=20, va="top", ha="left")
        subfig.suptitle(f"{expt} change in frequency of {panel['title']}-level {ext_type} events "
                        f"{period_phrase(period, gwl)}", fontsize=24, y=0.99)
        subfig.subplots_adjust(left=0.02, right=0.91, top=0.85, bottom=0.03,
                               wspace=0.05, hspace=0.02)

    maybe_save(fig, fig_dir, folder, file_name)
    return fig


def plot_ext_freq_with_model_diffs(
    exceedances, differences, agreement, model, expt, ext_type,
    diff_threshold, diff_vmin, diff_vmax, gwl=None, period=None,
    model_name=None, discrete=False, levels=None, levels_diff=None, n_cols=4,
    exclude_models=EXCLUDE, diff_label="(d)", diff_title=None,
    projection=None, suptitle=None, expt_name=None,
    fig_dir=None, folder=None, file_name=None,
):
    """
    plot_ext_freq (top, panels a-c) and plot_ext_freq_diff (bottom, panel d) in one figure.
    `model_name` and `expt_name` are unused (kept so old calls still work).
    """
    projection = projection or ccrs.Robinson()
    models = [m for m in differences if m not in exclude_models]
    n_rows_bottom = max(1, math.ceil(len(models) / n_cols))

    top_h, bottom_h = 10.0, n_rows_bottom * 3.0 + 1.0
    fig = plt.figure(figsize=(16, top_h + bottom_h))
    sf_top, sf_bottom = fig.subfigures(2, 1, height_ratios=[top_h, bottom_h])

    # ---- top: 3x3 frequency grid ----
    styles = _ext_freq_styles(ext_type, discrete, levels, levels_diff)
    axes_top = sf_top.subplots(3, 3, subplot_kw={"projection": projection}, squeeze=False)
    sf_top.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.14, wspace=0.08, hspace=0.12)
    _draw_ext_freq_grid(axes_top, exceedances, differences, agreement, model, expt, ext_type,
                        gwl, period, styles)

    for ax, symb in zip(axes_top[:, 0], EXT_TYPE_PARAMS[ext_type]["threshold_symbs"]):
        ax.annotate(symb, xy=(-0.05, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=20, rotation=90)
    label_panels(axes_top, y=1.10)

    _add_ext_freq_colorbars(
        sf_top, styles, ext_type,
        rects=([0.06, 0.07, 0.6, 0.022], [0.70, 0.07, 0.265, 0.022]),
        labels=("Frequency of occurrence\n(% of years within window)",
                "Change in frequency\n(difference in % of years)"),
    )

    # ---- bottom: per-model difference grid ----
    axes_bottom = sf_bottom.subplots(n_rows_bottom, n_cols, subplot_kw={"projection": projection},
                                     squeeze=False)
    sf_bottom.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05,
                              wspace=0.08, hspace=-0.35)
    mesh = _draw_model_diffs(axes_bottom, differences, models, expt, ext_type, diff_threshold,
                             gwl, period, diff_vmin, diff_vmax, title_fontsize=18)
    if mesh is not None:
        add_colorbar(sf_bottom, mesh, rect=[0.1, 0.06, 0.8, 0.04],
                     label="Change in frequency (difference in % of years)", extend="both")

    diff_title = diff_title or threshold_symb(ext_type, diff_threshold)
    sf_bottom.text(0.05, 0.85, diff_label, fontsize=20, va="bottom", ha="left")
    sf_bottom.text(0.5, 0.85,
                   f"{expt_name or expt} change in frequency of {diff_title}-level {ext_type} events "
                   f"{period_phrase(period, gwl)} by model",
                   fontsize=22, va="bottom", ha="center")

    if suptitle:
        fig.suptitle(suptitle, y=1.01, fontsize=22)
    maybe_save(fig, fig_dir, folder, file_name)
    return fig


# ---------------------------------------------------------------------------
# Recipes: timing and preindustrial statistics
# ---------------------------------------------------------------------------

def plot_hottest_coldest_month(extreme_months, extrema, title, n_cols=4, cmap=None):
    """
    Month of hottest/coldest occurrence, one map per model, averaged over all
    expt/member leaves. Accepts a DataTree or nested dict {model: {expt: {member: ds}}}.
    """
    cmap = cmap or plt.cm.twilight_shifted
    boundaries = np.arange(1, 14)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    models = list(getattr(extreme_months, "children", extreme_months).items())
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(16, n_rows * 3))

    for (model, node), ax in zip(models, axes.flat):
        da = average_leaves(node).sel(extrema=extrema).month
        mesh = plot_map_panel(ax, da, cmap, norm, title=model, title_fontsize=15)
    hide_unused_axes(axes, len(models))

    add_colorbar(fig, mesh, ax=axes, label="Month", fontsize=12, boundaries=boundaries,
                 ticks=np.arange(1.5, 13.5), ticklabels=MONTH_LABELS)
    fig.suptitle(title, fontsize=25, y=0.95)
    fig.tight_layout()
    return fig


def plot_pi_stats(ext_dict, variable, cmap, vmin, vmax, cbar_label, title, n_cols=4):
    """Preindustrial statistic, one map per model, averaged across ensemble members."""
    is_mean = "mean" in variable
    n_rows = math.ceil(len(ext_dict) / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(16, n_rows * 3))

    for (model, member_dict), ax in zip(ext_dict.items(), axes.flat):
        da = average_over_dim([ds[variable] for ds in member_dict.values()], dim="member_id")
        if is_mean:
            da = da - 273.15
        mesh = plot_map_panel(ax, da, cmap, vmin=vmin, vmax=vmax, title=model, title_fontsize=15)
    hide_unused_axes(axes, len(ext_dict))

    add_colorbar(fig, mesh, ax=axes, label=cbar_label, extend="both" if is_mean else "max")
    fig.suptitle(title, fontsize=25, y=0.95)
    fig.tight_layout()
    return fig