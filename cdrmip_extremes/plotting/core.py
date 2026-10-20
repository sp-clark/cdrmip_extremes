"""
core.py 
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

__all__ = [
    "MMM", "get_ds", "as_dataarray", "average_over_dim", "average_leaves",
    "average_leaf_datasets", "average_leaf_datasets_from_dict",
    "select_gwl", "period_phrase", "display_name", "mmm_first",
    "make_norm", "make_geo_grid", "plot_map_panel", "add_stippling",
    "add_colorbar", "label_panels", "hide_unused_axes", "save_figure", "maybe_save",
]

MMM = "multi_model_median"


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
    """Return a DataArray from a DataTree node, Dataset or DataArray."""
    ds = node.ds if hasattr(node, "ds") else node
    if not hasattr(ds, "data_vars"):
        return ds
    if varname is not None:
        return ds[varname]
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    raise ValueError(f"Dataset has multiple variables {data_vars}; pass `varname=...` to pick one.")


def average_over_dim(items, dim):
    """Concat a list of Datasets/DataArrays along a new dimension and average over it."""
    return xr.concat(items, dim=dim).mean(dim=dim)


def average_leaves(node, dim="leaf"):
    """Average all leaf datasets under a DataTree node or a {member: ds} dict."""
    if hasattr(node, "subtree"):
        leaves = [n.ds for n in node.subtree if n.has_data]
    else:
        leaves = list(node.values())
    return average_over_dim(leaves, dim)


# old names, kept so existing notebooks still work
average_leaf_datasets = average_leaves
average_leaf_datasets_from_dict = average_leaves


def select_gwl(da, gwl=None, period=None):
    """
    Pick the warming level: by value for crossing_years-based data
    (period None or 'gwl'), or the single entry of gwl_matching-based data
    (period 'final' / 'equivalent').
    """
    if period in (None, "gwl"):
        if gwl is None:
            raise ValueError("gwl is required unless period is 'final' or 'equivalent'.")
        return da.sel(gwl=gwl)
    return da.isel(gwl=0)


def period_phrase(period=None, gwl=None):
    """Title suffix describing the period, e.g. 'at 2.0°C'."""
    return {"final": "by end of simulation",
            "equivalent": "at equivalent GWL"}.get(period, f"at {gwl}°C")


def display_name(model):
    return "Multi-Model Median" if model == MMM else model


def mmm_first(models):
    """Model names with the multi-model median moved to the front."""
    models = list(models)
    return [m for m in models if m == MMM] + [m for m in models if m != MMM]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def make_norm(levels, cmap):
    """Resolve a cmap (name or object) and build a matching BoundaryNorm."""
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return cmap_obj, BoundaryNorm(levels, ncolors=cmap_obj.N, clip=True)


def make_geo_grid(n_rows, n_cols, figsize=None, projection=None, sharey=True, layout=None):
    """Create a fig/axes grid of cartopy map subplots (always 2D)."""
    return plt.subplots(
        n_rows, n_cols,
        subplot_kw={"projection": projection or ccrs.Robinson()},
        figsize=figsize or (16, n_rows * 10 / 3),
        sharey=sharey, squeeze=False, layout=layout,
    )


def plot_map_panel(ax, da, cmap, norm=None, vmin=None, vmax=None, title=None, title_fontsize=16,
                   wrap_cyclic=False, method="pcolormesh", extent=(0, 360, -90, 90)):
    """Plot a single map panel with coastlines. Returns the mesh/image (for colorbars)."""
    kw = dict(transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    if method == "imshow":
        mesh = ax.imshow(da, origin="lower", extent=extent, **kw)
    else:
        data, lon = da, da.lon
        if wrap_cyclic:
            data, lon = add_cyclic_point(np.asarray(da), coord=np.asarray(lon))
        mesh = ax.pcolormesh(lon, da.lat, data, **kw)
    ax.coastlines()
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    return mesh


def add_stippling(ax, da, hatch_color=(0, 0, 0, 0.5), hatches=("/////",)):
    """Overlay significance/agreement hatching on a single axis."""
    da = as_dataarray(da)
    data, lon = add_cyclic_point(np.asarray(da), coord=np.asarray(da.lon))
    return ax.contourf(
        lon, np.asarray(da.lat), data,
        transform=ccrs.PlateCarree(),
        levels=50, colors=[hatch_color], hatches=list(hatches), alpha=0,
    )


def add_colorbar(fig, mappable, rect=None, label=None, ax=None, fontsize=16,
                 tick_fontsize=14, ticks=None, ticklabels=None, **kwargs):
    """Add a colorbar, either in its own axes at `rect` or stolen from `ax`."""
    if rect is not None:
        kwargs.setdefault("orientation", "horizontal")
        cbar = fig.colorbar(mappable, cax=fig.add_axes(rect), **kwargs)
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


def label_panels(axes, labels=None, y=1.02):
    """Add (a)/(b)/(c)... labels above each axis in the top row."""
    top_row = axes[0]
    labels = labels or [f"({chr(97 + i)})" for i in range(len(top_row))]
    for ax, label in zip(top_row, labels):
        ax.text(0.0, y, label, transform=ax.transAxes, fontsize=20, va="bottom", ha="left")


def hide_unused_axes(axes, n_used):
    """Hide any axes in a grid beyond those actually used."""
    for ax in axes.flat[n_used:]:
        ax.set_visible(False)


def save_figure(fig, fig_dir, folder, file_name, dpi=300, bbox_inches="tight", pil_kwargs=None):
    """Save a figure to fig_dir/folder/file_name."""
    out_dir = Path(fig_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / file_name
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches,
                pil_kwargs=pil_kwargs or {"quality": 95})
    print(f"Saved: {path}")
    return path


def maybe_save(fig, fig_dir, folder, file_name):
    """Save only if all three location arguments were given."""
    if None not in (fig_dir, folder, file_name):
        save_figure(fig, fig_dir, folder, file_name)