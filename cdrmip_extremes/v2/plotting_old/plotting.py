"""
plotting.py
============
Shared utilities for climate map plots (extreme-frequency panels, GWL
difference maps, etc).

Organised in two layers:

1. Building blocks  - generic, dataset-agnostic helpers for setting up a grid
                       of cartopy map axes, plotting a panel, adding
                       colorbars, stippling, and panel labels.
2. Recipes          - the actual plot_* functions you call from notebooks,
                       built by composing the building blocks above with
                       dataset-specific extraction logic.

When you need a new plot, prefer writing a new "recipe" function that reuses
the building blocks rather than writing a new function from scratch.
"""

import math

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, Normalize
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def get_ds(data, *keys):
    """Walk into a nested dict or DataTree using the given keys and return a Dataset."""
    node = data
    for k in keys:
        node = node[k]
    return node.ds if hasattr(node, "ds") else node


def as_dataarray(node, varname=None):
    """Resolve a DataTree node / Dataset / DataArray down to a plottable DataArray.

    - DataTree node -> use its `.ds`
    - Dataset with `varname` given -> ds[varname]
    - Dataset with exactly one data variable -> that variable, auto-selected
    - Dataset with multiple variables and no `varname` -> raises, since it's
      ambiguous which one to plot
    - DataArray -> returned as-is
    """
    ds = node.ds if hasattr(node, "ds") else node
    if hasattr(ds, "data_vars"):  # it's a Dataset
        if varname is not None:
            return ds[varname]
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return ds[data_vars[0]]
        raise ValueError(
            f"Dataset has multiple variables {data_vars}; pass `varname=...` to pick one."
        )
    return ds  # already a DataArray


def average_over_dim(items, dim):
    """Concat a list of Datasets/DataArrays along a new dimension and average over it."""
    return xr.concat(items, dim=dim).mean(dim=dim)


def average_leaf_datasets(node, dim="leaf"):
    """Average over all leaf datasets under a DataTree node.

    Useful when a model node has children spanning e.g. experiment and
    ensemble member, and you want a single Dataset averaged across all of
    them.
    """
    leaf_datasets = [n.ds for n in node.subtree if n.has_data]
    return average_over_dim(leaf_datasets, dim)
    
def average_leaf_datasets_from_dict(model_dict):
    leaves = []

    for member, ds in model_dict.items():
        leaves.append(ds)

    # concatenate or average depending on your structure
    return xr.concat(leaves, dim="leaf").mean("leaf")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def make_norm(levels, cmap):
    """Resolve a cmap (name or object) and build a matching BoundaryNorm."""
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = BoundaryNorm(levels, ncolors=cmap_obj.N, clip=True)
    return cmap_obj, norm


def make_geo_grid(n_rows, n_cols, figsize=None, projection=None, sharey=True):
    """Create a fig/axes grid of cartopy map subplots (always 2D, even if 1 row/col)."""
    projection = projection or ccrs.Robinson()
    figsize = figsize or (16, n_rows * 10 / 3)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        subplot_kw={"projection": projection},
        figsize=figsize,
        sharey=sharey,
        squeeze=False,
    )
    return fig, axes


def plot_map_panel(ax, ds, cmap, norm=None, vmin=None, vmax=None, title=None, title_fontsize=16):
    """Plot a single pcolormesh map panel with coastlines. Returns the mesh (for colorbars).

    Pass either `norm` (e.g. from `make_norm`) for discrete/boundary colour
    scales, or `vmin`/`vmax` for a plain continuous scale. If `norm` is
    given, `vmin`/`vmax` are ignored (matplotlib's own behaviour).
    """
    lat, lon = ds.lat, ds.lon
    mesh = ax.pcolormesh(
        lon, lat, ds,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
    )
    ax.coastlines()
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    return mesh


def add_stippling(ax, agreement_ds, varname, hatch_color=(0, 0, 0, 0.5), hatches=("/////",)):
    """Overlay significance/agreement hatching on a single axis."""
    lon = np.array(agreement_ds.lon)
    lat = np.array(agreement_ds.lat)
    hatch_data, lon = add_cyclic_point(agreement_ds[varname], coord=lon)
    return ax.contourf(
        lon, lat, hatch_data,
        transform=ccrs.PlateCarree(),
        levels=50,
        colors=[hatch_color],
        hatches=list(hatches),
        alpha=0,
    )


def add_colorbar(fig, mappable, rect=None, label=None, ax=None, fontsize=16,
                  tick_fontsize=14, ticks=None, ticklabels=None, **kwargs):
    """Add a colorbar, either in a custom-positioned axis or attached to an
    existing set of axes.

    rect : [left, bottom, width, height] -> places the colorbar in its own
           axis at this position (use for precise custom layouts).
    ax   : existing Axes / array of Axes -> attaches a colorbar sized to
           span them (use for "one colorbar for the whole grid").
    Exactly one of `rect` / `ax` should be given.
    """
    if rect is not None:
        cbar_ax = fig.add_axes(rect)
        kwargs.setdefault("orientation", "horizontal")
        cbar = fig.colorbar(mappable, cax=cbar_ax, **kwargs)
    else:
        cbar = fig.colorbar(mappable, ax=ax, **kwargs)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    if label is not None:
        cbar.set_label(label, fontsize=fontsize)
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    return cbar


def label_panels(axes, labels=None):
    """Add (a)/(b)/(c)... labels above each axis in the top row."""
    top_row = axes[0]
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(top_row))]
    for ax, label in zip(top_row, labels):
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=20, va="bottom", ha="left")


def hide_unused_axes(axes, n_used):
    """Hide any axes in a grid beyond the number of plots actually drawn."""
    for ax in axes.flat[n_used:]:
        ax.set_visible(False)


# ---------------------------------------------------------------------------
# Recipe: extreme-frequency 3x3 grid (ramp up / ramp down / difference),
# with stippling for agreement and per-row sigma thresholds
# ---------------------------------------------------------------------------

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


def plot_ext_freq(ext_freq_data, differences, agreement, model, expt, varname, ext_type, model_name=None):
    """Plot ramp-up / ramp-down / difference frequency maps for each sigma threshold,
    with stippling for agreement on the difference column."""
    model_name = model_name or model
    params = EXT_TYPE_PARAMS[ext_type]

    exceedance = get_ds(ext_freq_data, model, expt, varname)
    diff = get_ds(differences, model, expt, varname)
    agreement_ds = get_ds(agreement, expt, varname)

    cmap1, norm1 = make_norm(params["levels"], params["cmap1"])
    cmap2, norm2 = make_norm(params["levels_diff"], params["cmap2"])

    fig, axes = make_geo_grid(3, 3, figsize=(16, 9))

    col_getters = [
        lambda t: exceedance[t].sel(period="pre"),
        lambda t: exceedance[t].sel(period="post"),
        lambda t: diff[t],
    ]

    for row, threshold in enumerate(THRESHOLDS):
        for col, getter in enumerate(col_getters):
            cmap, norm = (cmap2, norm2) if col == 2 else (cmap1, norm1)
            mesh = plot_map_panel(axes[row, col], getter(threshold), cmap, norm)
            if col == 0:
                mesh_main = mesh
            elif col == 2:
                mesh_diff = mesh
        axes[row, 0].annotate(
            params["threshold_symbs"][row], xy=(-0.025, 0.5), xycoords="axes fraction",
            ha="right", va="center", fontsize=20,
        )

    add_colorbar(
        fig, mesh_main, [0.057, -0.02, 0.6, 0.025],
        "Frequency of occurrence (% of years within window)",
    )
    add_colorbar(
        fig, mesh_diff, [0.7, -0.02, 0.265, 0.025],
        "Change in frequency (difference in % of years)",
        extend="both",
        ticks=params.get("diff_ticks"),
        ticklabels=params.get("diff_ticklabels"),
    )

    axes[0, 0].set_title("Ramp Up", fontsize=20)
    axes[0, 1].set_title("Ramp Down", fontsize=20)
    axes[0, 2].set_title("Ramp Down - Ramp Up", fontsize=20)

    for row, threshold in enumerate(THRESHOLDS):
        add_stippling(axes[row, 2], agreement_ds, threshold)

    label_panels(axes)
    fig.suptitle(
        f"{expt} {model_name} Change in {ext_type.capitalize()} Extreme Frequency at equivalent GWL",
        y=1.01, fontsize=25,
    )
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Recipe: extreme frequency plots
# ---------------------------------------------------------------------------

def plot_ext_freq_cont(ext_freq_data, differences, agreement, model, expt, varname, ext_type,
                        model_name=None, discrete=False, levels=None, levels_diff=None):
    """
    Plot a 3x3 grid: rows = sigma thresholds, columns = ramp up / ramp down /
    difference, with stippling for agreement on the difference column.

    ext_freq_data : DataTree or nested dict { model: { expt: { varname: ds } } }
                    raw frequency data, ds has 'sigma1'/'sigma2'/'sigma3' variables
                    with dims including period, lat, lon
    differences   : DataTree or nested dict { model: { expt: { varname: ds } } }
                    precomputed ramp_down - ramp_up, ds has 'sigma1'/'sigma2'/'sigma3'
                    variables with dims including lat, lon (no period dim)
    agreement     : DataTree or nested dict { expt: { varname: ds } }
                    ds has 'sigma1'/'sigma2'/'sigma3' variables with dims including
                    gwl, lat, lon (no model or branch dim)
    model, expt, varname : str, select which node to plot
    ext_type      : 'heat' or 'cold'
    model_name    : optional display name for the title (defaults to `model`)
    discrete      : bool, if True use BoundaryNorm colormaps (with under/over
                    extend colors on the difference panels) instead of
                    continuous Normalize colormaps
    levels        : array-like, optional override for the frequency-of-occurrence
                    boundaries (discrete mode) / range (continuous mode).
                    Defaults to EXT_TYPE_PARAMS[ext_type]['levels'].
    levels_diff   : array-like, optional override for the difference colorbar
                    boundaries/range. Defaults to EXT_TYPE_PARAMS[ext_type]['levels_diff'].
    """
    if model_name is None:
        model_name = model

    exceedance = get_ds(ext_freq_data, model, expt, varname)
    diff = get_ds(differences, model, expt, varname)
    agreement_ds = get_ds(agreement, expt, varname)

    to_plot = {}
    for threshold in THRESHOLDS:
        to_plot[threshold] = {
            "ramp_up": exceedance[threshold].sel(period="pre"),
            "ramp_down": exceedance[threshold].sel(period="post"),
            "difference": diff[threshold],
        }

    params = EXT_TYPE_PARAMS[ext_type]
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

    fig, axes = make_geo_grid(3, 3, figsize=(16, 9))

    for row_index, threshold in enumerate(THRESHOLDS):
        for col_index, key in enumerate(["ramp_up", "ramp_down", "difference"]):
            ax = axes[row_index, col_index]
            norm, cmap = (norm2, cmap2) if key == "difference" else (norm1, cmap1)
            plot_map_panel(ax, to_plot[threshold][key], cmap=cmap, norm=norm)
        axes[row_index, 0].annotate(
            params["threshold_symbs"][row_index], xy=(-0.025, 0.5), xycoords="axes fraction",
            ha="right", va="center", fontsize=20, rotation=0,
        )

    axes[0][0].set_title("Ramp Up", fontsize=20)
    axes[0][1].set_title("Ramp Down", fontsize=20)
    axes[0][2].set_title("Ramp Down - Ramp Up", fontsize=20)

    # stippling on the difference column
    for row_index, threshold in enumerate(THRESHOLDS):
        add_stippling(axes[row_index][2], agreement_ds, threshold)

    label_panels(axes, labels=["(a)", "(b)", "(c)"])

    add_colorbar(
        fig,
        plt.cm.ScalarMappable(norm=norm1, cmap=cmap1),
        rect=[0.057, -0.02, 0.6, 0.025],
        label="Frequency of occurrence (% of years within window)",
    )
    add_colorbar(
        fig,
        plt.cm.ScalarMappable(norm=norm2, cmap=cmap2),
        rect=[0.7, -0.02, 0.265, 0.025],
        label="Change in frequency (difference in % of years)",
        extend="both",
        ticks=params.get("diff_ticks"),
        ticklabels=params.get("diff_ticklabels"),
    )

    fig.suptitle(
        f"{expt} {model_name} Change in {ext_type.capitalize()} Extreme Frequency at equivalent GWL",
        y=1.01, fontsize=25,
    )
    fig.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Recipe: one map per model in a dict, same variable/expt
# ---------------------------------------------------------------------------

def plot_gwl_diff(data_dict, expt, varname=None, n_cols=3, levels=None, cmap=None,
                   cbar_label="SAT difference (°C)", suptitle=None):
    """Plot one difference map per model for a given experiment.

    data_dict : nested dict OR DataTree, structured as { model: { expt: node } }
    varname   : name of the data variable to plot, needed only if each
                model/expt node is a Dataset with more than one variable.
                Not needed if the node is already a DataArray, or a Dataset
                with a single variable.
    """
    to_plot = {}
    for model, expt_node in data_dict.items():
        if expt not in expt_node:
            continue
        to_plot[model] = as_dataarray(expt_node[expt], varname=varname)

    n_plots = len(to_plot)
    n_rows = math.ceil(n_plots / n_cols)

    levels = np.linspace(-1, 1, 21) if levels is None else levels
    cmap_resolved, norm = make_norm(levels, cmap or cm.RdBu_r)

    fig, axes = make_geo_grid(n_rows, n_cols)

    mesh = None
    for (model, ds), ax in zip(to_plot.items(), axes.flat):
        mesh = plot_map_panel(ax, ds, cmap_resolved, norm, title=model)

    hide_unused_axes(axes, n_plots)

    if mesh is not None:
        add_colorbar(fig, mesh, [0.1, -0.05, 0.8, 0.04], cbar_label, extend="both")

    fig.suptitle(suptitle or f"{expt} differences post vs pre-peak warming", fontsize=22)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Recipe: hottest/coldest month of occurrence, one map per model,
# averaged across all expt/member leaves under each model node
# ---------------------------------------------------------------------------

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_hottest_coldest_month(extreme_months, extrema, title, n_cols=4, cmap=None):
    """
    Supports:
      1) DataTree: { model: { expt: { member: ds } } }
      2) nested dict: same structure as above
    """

    cmap = cmap or plt.cm.twilight_shifted
    boundaries = np.arange(1, 14)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # ---- normalise input ----
    if hasattr(extreme_months, "children"):
        # DataTree case (as in your original code)
        models = list(extreme_months.children.items())
    else:
        # nested dict case
        models = list(extreme_months.items())

    n_plots = len(models)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(16, n_rows * 3))

    mesh = None

    for (model, model_dt), ax in zip(models, axes.flat):

        # ---- unify leaf aggregation step ----
        if hasattr(model_dt, "children"):
            combined = average_leaf_datasets(model_dt)
        else:
            # assume dict-of-dicts-of-datasets
            combined = average_leaf_datasets_from_dict(model_dt)

        da = combined.sel(extrema=extrema).month

        mesh = plot_map_panel(
            ax,
            da,
            cmap,
            norm,
            title=model,
            title_fontsize=15
        )

    hide_unused_axes(axes, n_plots)

    add_colorbar(
        fig, mesh, ax=axes, label="Month", fontsize=12,
        boundaries=boundaries,
        ticks=np.arange(1.5, 13.5),
        ticklabels=MONTH_LABELS,
    )

    fig.suptitle(title, fontsize=25, y=0.95)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Recipe: preindustrial statistic, one map per model, averaged across members
# ---------------------------------------------------------------------------

def plot_pi_stats(ext_dict, variable, cmap, vmin, vmax, cbar_label, title, n_cols=4):
    """Plot a preindustrial statistic, one map per model, averaged across
    ensemble members.

    ext_dict : nested dict, { model: { member: ds } }
    variable : name of the data variable to plot. Values are converted from
               Kelvin to Celsius if 'mean' appears in the variable name
               (mirrors the convention used for PI mean-temperature stats).
    """
    extend = "both" if "mean" in variable else "max"

    n_plots = len(ext_dict)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(16, n_rows * 3))

    mesh = None
    for (model, member_dict), ax in zip(ext_dict.items(), axes.flat):
        ds_list = [member_dict[member][variable] for member in member_dict]
        combined = average_over_dim(ds_list, dim="member_id")
        da = combined - 273.15 if "mean" in variable else combined
        mesh = plot_map_panel(ax, da, cmap, vmin=vmin, vmax=vmax, title=model, title_fontsize=15)

    hide_unused_axes(axes, n_plots)

    add_colorbar(fig, mesh, ax=axes, label=cbar_label, extend=extend)

    fig.suptitle(title, fontsize=25, y=0.95)
    fig.tight_layout()
    return fig