"""
plotting.py
"""

import math

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm, CenteredNorm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.gridspec as gridspec


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


def plot_map_panel(ax, ds, cmap, norm=None, vmin=None, vmax=None, title=None, title_fontsize=16,
                    wrap_cyclic=False, method="pcolormesh", extent=(0, 360, -90, 90)):
    """Plot a single map panel with coastlines. Returns the mesh/image (for colorbars).

    Pass either `norm` (e.g. from `make_norm`) for discrete/boundary colour
    scales, or `vmin`/`vmax` for a plain continuous scale. If `norm` is
    given, `vmin`/`vmax` are ignored (matplotlib's own behaviour).

    method : {'pcolormesh', 'imshow'}
        'pcolormesh' (default) does an exact per-cell transform via
        cartopy. 'imshow' instead warps the whole array as a single image
        -- coarser, but immune to the cartopy pcolormesh seam-wrapping bug
        (see wrap_cyclic below), which tends to surface on fields with
        NaNs near the antimeridian (e.g. multi-model medians). `extent` is
        only used in 'imshow' mode.

    wrap_cyclic : bool
        Only applies to 'pcolormesh'. If True, close the longitude seam
        with cartopy's add_cyclic_point before plotting -- needed for grids
        that don't explicitly reach 360 (e.g. a CDO-style 0..358 grid).
        Without this, cartopy's pcolormesh seam-wrapping can occasionally
        raise "TypeError: 'GeometryCollection' object is not subscriptable".
        Off by default since most existing recipes don't need it.
    """
    if method == "imshow":
        plot_data = ds
        mesh = ax.imshow(
            plot_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
            origin="lower", extent=extent,
        )
    else:
        lat, lon = ds.lat, ds.lon
        plot_data = ds
        if wrap_cyclic:
            plot_data, lon = add_cyclic_point(np.asarray(ds), coord=np.asarray(lon))
        mesh = ax.pcolormesh(
            lon, lat, plot_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        )
    ax.coastlines()
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    return mesh


def add_stippling(ax, da, hatch_color=(0, 0, 0, 0.5), hatches=("/////",)):
    """Overlay significance/agreement hatching on a single axis.

    da : a single 2D (lat, lon) DataArray -- already selected down to one
    threshold/gwl/etc by the caller (unlike the old version, this no longer
    takes a Dataset + varname, since callers now generally already have a
    plain DataArray in hand after indexing).
    """
    lon = np.array(da.lon)
    lat = np.array(da.lat)
    hatch_data, lon = add_cyclic_point(np.asarray(da), coord=lon)
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


def save_figure(fig, fig_dir, folder, file_name, dpi=300, bbox_inches="tight",pil_kwargs={"quality": 95}):
    """
    Save a figure to fig_dir/folder/file_name, creating the folder if needed.

    folder : {'main', 'supplementary'} (or any other subfolder name you use)
    file_name : str, including extension, e.g. 'figure1.png'
    """
    from pathlib import Path
    out_dir = Path(fig_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / file_name
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches,pil_kwargs=pil_kwargs)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Recipe: extreme-frequency 3x3 grid (ramp up / ramp down / difference)
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


def plot_ext_freq_diff(differences, expt, ext_type, threshold, levels, label, title,
                     vmin, vmax, gwl=None, period=None,
                     exclude_models=("multi_model_median", "Multi-Model Median"),
                     n_cols=4, fig_dir=None, folder=None, file_name=None):
    """
    Plot a per-model difference map (e.g. branch_differences: ramp_down -
    ramp_up) at a given GWL/threshold, one panel per model in a grid.
 
    differences : nested dict { model: { expt: { ext_type: ds } } }
                  ds has sigma1/sigma2/sigma3, dims (gwl, lat, lon)
    expt, ext_type, threshold : select which data to plot
    gwl           : float, optional. The global warming level to select via
                    .sel(gwl=gwl) -- for crossing_years-based data where all
                    models share the same fixed GWL targets.
    period        : optional, any truthy value. If set, selects via
                    .isel(gwl=0) -- for gwl_matching-based data where each
                    model has a single but differently-labelled gwl entry.
                    Exactly one of gwl / period should be given.
    levels        : currently unused for colour mapping (continuous vmin/vmax
                    is used) -- kept in case you want to switch to discrete
                    BoundaryNorm later.
    label         : top-left corner label (e.g. a panel letter/identifier)
    title         : short name for the variable being plotted, used in suptitle
    exclude_models : model keys to skip
    n_cols        : number of columns in the model grid (rows auto-computed)
    fig_dir, folder, file_name : optional save path
 
    Uses imshow rather than pcolormesh -- deliberate, since multi-model/
    difference fields are exactly where the cartopy pcolormesh seam-wrapping
    bug tends to surface with NaNs.
    """
    if gwl is None and period is None:
        raise ValueError("Provide either gwl (crossing_years-based) or period (gwl_matching-based).")
    if gwl is not None and period is not None:
        raise ValueError("Provide only one of gwl / period, not both.")
 
    def _select(da):
        return da.isel(gwl=0) if period is not None else da.sel(gwl=gwl)
 
    params = EXT_TYPE_PARAMS[ext_type]
    symb = params["threshold_symbs"][THRESHOLDS.index(threshold)]
 
    if ext_type == "heat":
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_extremes(under="midnightblue", over="maroon")
    elif ext_type == "cold":
        cmap = plt.cm.RdBu.copy()
        cmap.set_extremes(under="maroon", over="midnightblue")
    else:
        raise ValueError(f"Unknown ext_type: {ext_type}")
 
    to_plot = {
        model: _select(get_ds(differences, model, expt, ext_type)[threshold])
        for model in differences.keys()
        if model not in exclude_models
    }
 
    n_plots = len(to_plot)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(17, n_rows * 2.85))
 
    mesh = None
    for (model, da), ax in zip(to_plot.items(), axes.flat):
        mesh = plot_map_panel(
            ax, da, cmap, vmin=vmin, vmax=vmax, title=model, title_fontsize=16,
            method="imshow",
        )
 
    hide_unused_axes(axes, n_plots)
 
    add_colorbar(
        fig, mesh,
        rect=[1.01, 0.06, 0.01, 0.8],
        label="Difference in % of years",
        fraction=0.02, aspect=20, orientation="vertical",
        extend="both", tick_fontsize=14,
    )
 
    fig.text(0, 0.95, label, fontsize=20, va="bottom", ha="left")
    if period == 'final':
        fig.suptitle(
            f"{expt} change in frequency of {title}-level {ext_type} events by end of simulation",
            fontsize=24, y=1.0,
        )
    elif period == 'equivalent':
        fig.suptitle(
            f"{expt} change in frequency of {title}-level {ext_type} events at equivalent GWL",
            y=1.01, fontsize=25,
        )
    elif gwl is not None:
        fig.suptitle(
            f"{expt} change in frequency of {title}-level {ext_type} events at {gwl}°C",
            fontsize=24, y=1.0,
        )
    fig.tight_layout()
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    plt.show()
    return fig

def plot_ext_freq_diff_corr_all_thresholds(differences, expt, ext_type, levels, title,
                     vmin, vmax, gwl=None, period=None,
                     exclude_models=("multi_model_median", "Multi-Model Median"),
                     n_cols=3, fig_dir=None, folder=None, file_name=None,
                     corr_dict=None):
    """
    Like `plot_ext_freq_diff_corr`, but for a single ext_type plots all
    three sigma thresholds in one figure instead of just one.
 
    Layout: one 2-row block per threshold (sigma1 on top, then sigma2, then
    sigma3), each block using `n_cols` columns of model panels -- so the
    full grid is 6 x n_cols, and each threshold can show up to 2 * n_cols
    models (n_cols=3 -> 6 models/threshold).
 
    differences, expt, ext_type, label, title, vmin, vmax, gwl, period,
    exclude_models, fig_dir, folder, file_name, corr_dict : see
    `plot_ext_freq_diff_corr` -- identical meaning, just no `threshold` arg
    since all three thresholds are plotted here.
    levels : currently unused (continuous vmin/vmax is used) -- kept for
              parity with `plot_ext_freq_diff_corr`.
    n_cols : columns per threshold block (default 3). Each block is always
             exactly 2 rows tall.
    """
    if gwl is None and period is None:
        raise ValueError("Provide either gwl (crossing_years-based) or period (gwl_matching-based).")
    if gwl is not None and period is not None:
        raise ValueError("Provide only one of gwl / period, not both.")
 
    def _select(da):
        return da.isel(gwl=0) if period is not None else da.sel(gwl=gwl)
 
    def _panel_title(model, threshold):
        if corr_dict is None:
            return model
        try:
            corr_val = corr_dict[model][ext_type]
        except KeyError:
            return model
        if hasattr(corr_val, "dims") and "threshold" in corr_val.dims:
            corr_val = corr_val.sel(threshold=threshold)
        elif np.ndim(corr_val) > 0:
            corr_val = corr_val[THRESHOLDS.index(threshold)]
        if model == 'multi_model_median':
            model_name = 'Multi-Model Median'
        else:
            model_name=model
        return f"{model_name}, corr={float(corr_val):.2f}"
 
    params = EXT_TYPE_PARAMS[ext_type]
 
    if ext_type == "heat":
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_extremes(under="midnightblue", over="maroon")
    elif ext_type == "cold":
        cmap = plt.cm.RdBu.copy()
        cmap.set_extremes(under="maroon", over="midnightblue")
    else:
        raise ValueError(f"Unknown ext_type: {ext_type}")
 
    rows_per_threshold = 2
    n_rows = rows_per_threshold * len(THRESHOLDS)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=(n_cols * 5.7, n_rows * 2.85))
 
    mesh = None
    for t_index, threshold in enumerate(THRESHOLDS):
        to_plot = {
            model: _select(get_ds(differences, model, expt, ext_type)[threshold])
            for model in differences.keys()
            if model not in exclude_models
        }
        block_axes = axes[t_index * rows_per_threshold:(t_index + 1) * rows_per_threshold, :]
 
        for (model, da), ax in zip(to_plot.items(), block_axes.flat):
            mesh = plot_map_panel(
                ax, da, cmap, vmin=vmin, vmax=vmax,
                title=_panel_title(model, threshold), title_fontsize=20,
                method="imshow",
            )
 
        hide_unused_axes(block_axes, len(to_plot))
 
        block_axes[0, 0].annotate(
            params["threshold_symbs"][t_index], xy=(-0.05, -0.1), xycoords="axes fraction",
            ha="right", va="center", fontsize=30, rotation=0,
        )
 
    add_colorbar(
        fig, mesh,
        rect=[1.0, 0.0, 0.02, 0.95],
        label="Difference in % of years",
        fraction=0.02, aspect=20, orientation="vertical",
        extend="both", tick_fontsize=14,
    )
 
    if period == 'final':
        fig.suptitle(
            f"{expt} change in frequency of extreme {ext_type} events by end of simulation",
            fontsize=24, y=1.0,
        )
    elif period == 'equivalent':
        fig.suptitle(
            f"{expt} change in frequency of extreme {ext_type} events at equivalent GWL",
            y=1.01, fontsize=25,
        )
    elif period == 'gwl':
        fig.suptitle(
            f"{expt} change in frequency of extreme {ext_type} events at {gwl}°C",
            fontsize=24, y=1.0,
        )
    fig.tight_layout()
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    plt.show()
    return fig

def plot_ext_freq(exceedances, differences, agreement, model, expt, ext_type, gwl=None, period=None,
                        model_name=None, discrete=False, levels=None, levels_diff=None,
                        wrap_cyclic=True, fig_dir=None, folder=None, file_name=None,
                         expt_name=None, suptitle=None):
    """
    Plot a 3x3 grid: rows = sigma thresholds, columns = ramp up / ramp down /
    difference, with stippling for agreement on the difference column.

    exceedances : nested dict { model: { expt: { ext_type: ds } } }
                  ds has 'sigma1'/'sigma2'/'sigma3', dims (branch, gwl, lat, lon)
    differences : nested dict { model: { expt: { ext_type: ds } } }
                  ramp_down - ramp_up, dims (gwl, lat, lon)
    agreement   : nested dict { expt: { ext_type: ds } }
                  dims (gwl, lat, lon)
    model, expt, ext_type : str, select which node to plot
    gwl           : float, optional. The global warming level to select via
                    .sel(gwl=gwl) -- used for crossing_years-based data,
                    where every model shares the same fixed GWL targets
                    (1.5/2.0/3.0 etc).
    period        : optional, any truthy value. If set, selects via
                    .isel(gwl=0) instead of gwl -- used for gwl_matching-
                    based data, where each model/member matched a
                    *different* GSAT level (its own late-period mean), so
                    there's no single gwl value valid across all of them;
                    each dataset just has one (differently-labelled) gwl
                    entry, and isel(gwl=0) picks "the only one there is"
                    regardless of its value. Exactly one of `gwl`/`period`
                    should be given.
    model_name    : optional display name for the title (defaults to `model`)
    discrete      : bool, if True use BoundaryNorm colormaps (with under/over
                    extend colors on the difference panels) instead of
                    continuous Normalize colormaps
    levels        : array-like, optional override for the frequency-of-occurrence
                    boundaries (discrete mode) / range (continuous mode).
                    Defaults to EXT_TYPE_PARAMS[ext_type]['levels'].
    levels_diff   : array-like, optional override for the difference colorbar
                    boundaries/range. Defaults to EXT_TYPE_PARAMS[ext_type]['levels_diff'].
    wrap_cyclic   : bool, default True -- see plot_map_panel's docstring.
                    Set False if your grid already explicitly closes at 360.
    fig_dir, folder, file_name : optional -- if all three are given, the
                    figure is saved to fig_dir/folder/file_name (folder is
                    typically 'main' or 'supplementary', but any subfolder
                    name works). Folder is created if needed.
    """
    if gwl is None and period is None:
        raise ValueError("Provide either gwl (crossing_years-based) or period (gwl_matching-based).")
    # if gwl is not None and period is not None:
    #     raise ValueError("Provide only one of gwl / period, not both.")

    if model_name is None:
        model_name = model
    if expt_name is None:
        expt_name = expt

    def _select(da):
        return da.isel(gwl=0) if period is not None else da.sel(gwl=gwl)

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

    fig, axes = make_geo_grid(3, 3, figsize=(16, 10))

    for row_index, threshold in enumerate(THRESHOLDS):
        for col_index, key in enumerate(["ramp_up", "ramp_down", "difference"]):
            ax = axes[row_index, col_index]
            norm, cmap = (norm2, cmap2) if key == "difference" else (norm1, cmap1)
            plot_map_panel(ax, to_plot[threshold][key], cmap=cmap, norm=norm, method='imshow', wrap_cyclic=False)
        axes[row_index, 0].annotate(
            params["threshold_symbs"][row_index], xy=(-0.025, 0.5), xycoords="axes fraction",
            ha="right", va="center", fontsize=20, rotation=0,
        )
    if expt == 'ssp534-over':
        axes[0][0].set_title("pre-overshoot", fontsize=20)
        axes[0][1].set_title("post-overshoot", fontsize=20)
        axes[0][2].set_title("difference (post-pre)", fontsize=20)
    elif expt == '1pctCO2-cdr':
        axes[0][0].set_title("Ramp Up", fontsize=20)
        axes[0][1].set_title("Ramp Down", fontsize=20)
        axes[0][2].set_title("Ramp Down - Ramp Up", fontsize=20)
    # stippling on the difference column
    for row_index, threshold in enumerate(THRESHOLDS):
        add_stippling(axes[row_index][2], _select(agreement_ds[threshold]))

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
    
    if suptitle:
        fig.suptitle(suptitle,y=1.01, fontsize=25)
    elif period == 'final':
        fig.suptitle(
            f"{expt_name} {model_name} Change in {ext_type.capitalize()} Extreme Frequency\nby End of Simulation",
            y=1.01, fontsize=25,
        )
    elif period == 'equivalent':
        fig.suptitle(
            f"{expt_name} {model_name} Change in {ext_type.capitalize()} Extreme Frequency",
            y=1.01, fontsize=25,
        )
    elif period == 'gwl':
        fig.suptitle(
            f"{expt_name} {model_name} Change in {ext_type.capitalize()} Extreme Frequency\n at {gwl}°C",
            y=1.01, fontsize=25,
        )
    fig.tight_layout()
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    return fig


# ---------------------------------------------------------------------------
# Recipe: one map per model in a dict, same variable/expt
# ---------------------------------------------------------------------------

def plot_gwl_diff(data_dict, expt, varname=None, n_cols=3, levels=None, cmap=None,
                   cbar_label="SAT difference (°C)", suptitle=None, agreement=None):
    """...
    agreement : optional 2D (lat, lon) DataArray; if given, stippled onto the
                multi-model median panel via add_stippling.
    """
    to_plot = {}
    for model, expt_node in data_dict.items():
        if expt not in expt_node:
            continue
        to_plot[model] = as_dataarray(expt_node[expt], varname=varname)
    
    # put multi_model_median first
    if 'multi_model_median' in to_plot:
        to_plot = {'multi_model_median': to_plot['multi_model_median'],
                   **{k: v for k, v in to_plot.items() if k != 'multi_model_median'}}

    n_plots = len(to_plot)
    n_rows = math.ceil(n_plots / n_cols)

    levels = np.linspace(-1, 1, 21) if levels is None else levels
    cmap_resolved, norm = make_norm(levels, cmap or cm.RdBu_r)

    fig, axes = make_geo_grid(n_rows, n_cols)

    mesh = None
    for (model, ds), ax in zip(to_plot.items(), axes.flat):
        if model == 'multi_model_median':
            model_name = 'Multi-Model Median'
        else:
            model_name = model
        mesh = plot_map_panel(ax, ds, cmap_resolved, norm, title=model_name, method='imshow')
        if agreement is not None and model == 'multi_model_median':
            add_stippling(ax, agreement)

    hide_unused_axes(axes, n_plots)

    if mesh is not None:
        add_colorbar(fig, mesh, [0.1, -0.025, 0.8, 0.02], cbar_label, extend="both")

    fig.suptitle(suptitle or f"{expt} differences post vs pre-peak warming", fontsize=22, y =1.02)
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

# ---------------------------------------------------------------------------
# Storyline Plot
# -------------------------------------------------------------------------

def _draw_grid(subfig, items, amoc_changes, cmap, levels,
               cbar_label, panel_label, title):
    """Draw one 2x4 map grid with its own colorbar, suptitle and panel label."""
    axes = subfig.subplots(2, 4, subplot_kw={"projection": ccrs.NorthPolarStereo()})
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    for (model, da), ax in zip(items, axes.flat):
        mesh = ax.pcolormesh(da.lon, da.lat, da,
                             transform=ccrs.PlateCarree(),
                             cmap=cmap, norm=norm, shading='auto')
        ax.set_title(f"{model}, $\\Delta$ = {amoc_changes[model]:.2f} Sv", fontsize=16)
        ax.set_extent([-100, 60, 30, 90], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=False)
    cbar_ax = subfig.add_axes([0.92, 0.08, 0.012, 0.72])
    cbar = subfig.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16)
    subfig.text(0.01, 0.98, panel_label, fontsize=20, va='top', ha='left')
    subfig.suptitle(title, fontsize=25, y=0.99)
    subfig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.02,
                           wspace=0.05, hspace=0.05)
 
 
def plot_freq_and_intens_amoc_stereo(ext_type, freq_differences, intens_differences,
                                     gwl, threshold, freq_levels, intens_levels,
                                     amoc_changes,period=None):
    """Stacked figure: (a) exceedance-frequency change, (b) extreme-month SAT change.
 
    ext_type: 'Heat' or 'Cold' (intensity panel uses ext_type.lower()).
    """
    heat_symbs = {'sigma1': '1$\\sigma$', 'sigma2': '2$\\sigma$', 'sigma3': '3$\\sigma$'}
    cold_symbs = {'sigma1': '-0.25$\\sigma$', 'sigma2': '-0.5$\\sigma$', 'sigma3': '-1$\\sigma$'}
 
    if ext_type == 'Heat':
        symb, freq_cmap = heat_symbs[threshold], 'RdBu_r'
    else:
        symb, freq_cmap = cold_symbs[threshold], 'RdBu'
 
    fig = plt.figure(figsize=(16, 13))
    top, bottom = fig.subfigures(2, 1, hspace=0.02)
    if period == 'final':
        freq_items = [(m, freq_differences[m][threshold].isel(gwl=0))
                      for m in freq_differences]
        _draw_grid(top, freq_items, amoc_changes, freq_cmap, freq_levels,
                   "Difference in % of years", '(a)',
                   f"Change in {symb} {ext_type} Extreme Exceedance Frequency by End of Simulation")
        intens_items = [(m, ds.isel(gwl=0).sel(extrema=ext_type.lower()).tas)
                        for m, ds in intens_differences.items()]
        _draw_grid(bottom, intens_items, amoc_changes, 'RdBu_r', intens_levels,
                   "SAT Anomaly ($^o$C)", '(b)',
                   f"Change in {ext_type.capitalize()}est Month Temperature by End of Simulation")
    elif period == 'gwl':
        freq_items = [(m, freq_differences[m][threshold].sel(gwl=gwl))
                      for m in freq_differences]
        _draw_grid(top, freq_items, amoc_changes, freq_cmap, freq_levels,
                   "Difference in % of years", '(a)',
                   f"Change in {symb} {ext_type} Extreme Exceedance Frequency at {gwl}°C GWL")
        intens_items = [(m, ds.sel(gwl=gwl, extrema=ext_type.lower()).tas)
                        for m, ds in intens_differences.items()]
        _draw_grid(bottom, intens_items, amoc_changes, 'RdBu_r', intens_levels,
                   "SAT Anomaly ($^o$C)", '(b)',
                   f"Change in {ext_type.capitalize()}est Month Temperature at {gwl}°C GWL")
 
    plt.show()
    return fig
 
def _draw_2x2_robinson(subfig, rows, panel_label):
    """One 2x2 Robinson grid; `rows` = two dicts, each with keys:
    data ({title: DataArray}), cmap, vmin, vmax, cbar_label."""
    axes = subfig.subplots(2, 2, subplot_kw={'projection': ccrs.Robinson()})
    cbar_boxes = [[0.93, 0.52, 0.012, 0.38], [0.93, 0.06, 0.012, 0.38]]
    for row, ax_row, box in zip(rows, axes, cbar_boxes):
        for (name, da), ax in zip(row['data'].items(), ax_row):
            im = ax.imshow(da, transform=ccrs.PlateCarree(),
                           cmap=row['cmap'], vmin=row['vmin'], vmax=row['vmax'],
                           origin='lower', extent=(0, 360, -90, 90))
            ax.coastlines()
            ax.set_title(name, fontsize=18)
        cbar_ax = subfig.add_axes(box)
        cbar = subfig.colorbar(im, cax=cbar_ax, extend='both')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(row['cbar_label'], fontsize=16)
    subfig.text(0.005, 0.99, panel_label, fontsize=20, va='top', ha='left')
    subfig.subplots_adjust(left=0.02, right=0.91, top=0.92, bottom=0.03,
                           wspace=0.05, hspace=0.18)
 
 
def plot_mean_freq_and_tas_stacked(freq_rows, tas_rows):
    """(a) frequency 2x2 block on top, (b) SAT 2x2 block below."""
    fig = plt.figure(figsize=(16, 18))
    top, bottom = fig.subfigures(2, 1, hspace=0.02)
    _draw_2x2_robinson(top, freq_rows, '(a)')
    _draw_2x2_robinson(bottom, tas_rows, '(b)')
    plt.show()
    return fig

def plot_ext_freq_diff_stacked(differences, expt, ext_type, panels,
                               vmin, vmax, gwl=None, period=None,
                               exclude_models=("multi_model_median", "Multi-Model Median"),
                               n_cols=4, fig_dir=None, folder=None, file_name=None):
    """
    Stack two (or more) plot_ext_freq_diff grids vertically in one figure,
    one subfigure per entry in `panels`, each with its own suptitle,
    corner label and colorbar.
 
    panels : list of dicts, one per stacked grid, each with keys:
             'threshold' (e.g. 'sigma1'), 'label' (e.g. '(a)'),
             'title' (short name used in that grid's suptitle).
    All other arguments are shared across the panels and behave exactly as
    in plot_ext_freq_diff.
    """
    if gwl is None and period is None:
        raise ValueError("Provide either gwl (crossing_years-based) or period (gwl_matching-based).")
    if gwl is not None and period is not None:
        raise ValueError("Provide only one of gwl / period, not both.")
 
    def _select(da):
        return da.isel(gwl=0) if period is not None else da.sel(gwl=gwl)
 
    if ext_type == "heat":
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_extremes(under="midnightblue", over="maroon")
    elif ext_type == "cold":
        cmap = plt.cm.RdBu.copy()
        cmap.set_extremes(under="maroon", over="midnightblue")
    else:
        raise ValueError(f"Unknown ext_type: {ext_type}")
 
    models = [m for m in differences.keys() if m not in exclude_models]
    n_rows = math.ceil(len(models) / n_cols)
 
    fig = plt.figure(figsize=(17, len(panels) * (n_rows * 2.45 + 1.0)))
    subfigs = np.atleast_1d(fig.subfigures(len(panels), 1, hspace=0.03))
 
    for subfig, panel in zip(subfigs, panels):
        axes = subfig.subplots(n_rows, n_cols,
                               subplot_kw={"projection": ccrs.Robinson()},
                               squeeze=False)
        mesh = None
        for model, ax in zip(models, axes.flat):
            da = _select(get_ds(differences, model, expt, ext_type)[panel["threshold"]])
            mesh = plot_map_panel(ax, da, cmap, vmin=vmin, vmax=vmax,
                                  title=model, title_fontsize=16, method="imshow")
        hide_unused_axes(axes, len(models))
 
        add_colorbar(subfig, mesh, rect=[0.93, 0.10, 0.01, 0.70],
                     label="Difference in % of years",
                     orientation="vertical", extend="both", tick_fontsize=14)
 
        subfig.text(0.005, 0.97, panel["label"], fontsize=20, va="top", ha="left")
        title = panel["title"]
        if period == "final":
            sup = f"{expt} change in frequency of {title}-level {ext_type} events by end of simulation"
        elif period == "equivalent":
            sup = f"{expt} change in frequency of {title}-level {ext_type} events at equivalent GWL"
        else:
            sup = f"{expt} change in frequency of {title}-level {ext_type} events at {gwl}°C"
        subfig.suptitle(sup, fontsize=24, y=0.99)
        subfig.subplots_adjust(left=0.02, right=0.91, top=0.85, bottom=0.03,
                               wspace=0.05, hspace=0.02)
 
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Recipe: GSAT timeseries with GWL-matching year markers
# ---------------------------------------------------------------------------
 
def plot_gsat_timeseries(gsat_rolling, gwl_matching_years, colour_dict,
                         title=None,
                         ylabel="GSAT (°C)", xlim=(1850, 2300), ylim=(-0.2, 4.1),
                         end_marker_year=2190, figsize=(16, 8), legend_ncol=5,
                         fig_dir=None, folder=None, file_name=None):
    """
    Timeseries of rolling GSAT per model/experiment, with dashed lines at each
    model's matched GWL, an up-triangle at the ramp-up GWL-matching year and a
    down-triangle at `end_marker_year` (end of simulation).
 
    gsat_rolling : nested dict { model: { expt: ds } }
                   ds has a 'gsat_rolling' variable with a time dim
    gwl_matching_years : nested dict { model: { expt: { member: ds } } }
                   ds has 'gwl_matching_years' (with a 'branch' dim covering
                   'ramp_up'/'ramp_down') and a scalar 'gwl'; the median is
                   taken across members
    colour_dict : { model: colour }
    end_marker_year : x-position for the end-of-simulation (down-triangle)
                   marker
    fig_dir, folder, file_name : optional -- if all three are given, the
                   figure is saved via save_figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
 
    for model, expt_dt in gsat_rolling.items():
        for expt, da in expt_dt.items():
            da.gsat_rolling.plot(
                ax=ax, label=model, linewidth=3, color=colour_dict[model],
            )
 
    for model, expt_dict in gwl_matching_years.items():
        for expt, member_dt in expt_dict.items():
            members = [ds.drop_vars('gwl') for ds in member_dt.values()]
            median_ds = xr.concat(members, dim='member').median(dim=['member'])
            gwl = np.round(np.median([ds.gwl.item() for ds in member_dt.values()]), 2)
            pre_gwl_year = median_ds.sel(branch='ramp_up').gwl_matching_years.values
            ax.axhline(y=gwl, color=colour_dict[model], linestyle='--', alpha=0.5)
            ax.scatter(end_marker_year, gwl, color=colour_dict[model],
                       marker='v', s=400, zorder=5)
            ax.scatter(pre_gwl_year, gwl, color=colour_dict[model],
                       marker='^', s=400, zorder=5)
 
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
 
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), unique.keys(),
        loc='upper center', bbox_to_anchor=(0.5, -0.12),
        ncol=legend_ncol, fontsize=18, frameon=False,
    )
    ax.set_title(title, fontsize=24, y=1.02)
    fig.tight_layout()
 
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    return fig


# ---------------------------------------------------------------------------
#  local peak-warming year relative to GSAT peak, one map per model
# ---------------------------------------------------------------------------
 
def plot_peak_warming_years(peak_years, peak_gsat, expt, n_cols=3, levels=None,
                            cmap=None, figsize=None,
                            cbar_label="Local peak warming year rel. to GSAT peak warming year",
                            suptitle=None, fig_dir=None, folder=None, file_name=None):
    """
    """
    models = [m for m in peak_years.keys() if expt in peak_years[m]]
    if 'multi_model_median' in models:
        models = ['multi_model_median'] + [m for m in models if m != 'multi_model_median']
 
    levels = np.arange(-40, 41, 10) if levels is None else levels
    cmap_resolved, norm = make_norm(levels, cmap or cm.RdBu_r)
 
    n_plots = len(models)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=figsize or (15, n_rows * 10 / 3))
 
    mesh = None
    for model, ax in zip(models, axes.flat):
        peak_gsat_year = peak_gsat[model]['peak_year']
        da = peak_years[model][expt] - peak_gsat_year
        model_name = 'Multi-Model Median' if model == 'multi_model_median' else model
        mesh = plot_map_panel(
            ax, da, cmap_resolved, norm=norm, method='imshow',
            title=f'{model_name} | Peak GSAT year: {peak_gsat_year}',
        )
 
    hide_unused_axes(axes, n_plots)
 
    add_colorbar(
        fig, mesh, rect=[0.1, -0.025, 0.8, 0.03],
        label=cbar_label,
        fraction=0.02, aspect=20, orientation="horizontal",
        tick_fontsize=14, extend='both',
    )
 
    fig.suptitle(suptitle or f"{expt} year of peak warming", fontsize=22, y=1.02)
    fig.tight_layout()
 
    if fig_dir is not None and folder is not None and file_name is not None:
        save_figure(fig, fig_dir, folder, file_name)
    return fig


# ---------------------------------------------------------------------------
#  local reversibility relative to gsat reversibility
# ---------------------------------------------------------------------------

def plot_reversibility_rel_gsat_normalised(reversibility,reversibility_gsat,window,bins,rev_year,expt,CO2):

    fig, axes = make_geo_grid(3,3,figsize=(15,10))

    # put multi_model_median first
    models = ['multi_model_median'] + [m for m in reversibility.keys() if m != 'multi_model_median']
    
    for model, ax in zip(models,axes.flat):
        da = reversibility[model][expt].sel(year=rev_year)
        gsat_da = np.round(reversibility_gsat[model][expt].sel(year=rev_year).values,2)
        da = da - gsat_da
        levels = np.linspace(0-window,0+window,bins)
        cmap_resolved, norm = make_norm(levels,cm.RdBu_r)
        if model == 'multi_model_median':
            title = f'Multi-Model Median | GSAT Difference: {gsat_da}°C'
        else:
            title = f'{model} | GSAT Difference: {gsat_da}°C'
        im = plot_map_panel(
            ax,
            da,
            cmap=cmap_resolved,
            norm=norm,
            method='imshow',
            title=title
        )
    
    add_colorbar(
        fig, im, rect=[0.1, -0.025, 0.8, 0.03],
        label="Local SAT difference rel. to GSAT difference (°C)",
        fraction=0.02, aspect=20, orientation="horizontal",
        tick_fontsize=14, extend='both'
    )
        
    co2_value = int(np.round(CO2.sel(year=rev_year+100).values,0))
    fig.suptitle(
        f"SAT Reversibility following return to {co2_value}ppm CO₂ (Year {280-rev_year})",
        fontsize=22, y=1.02
    )
    fig.tight_layout()
    plt.show()
    return fig