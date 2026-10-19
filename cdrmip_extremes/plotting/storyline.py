"""
storyline.py -- multi-panel storyline figures: North Atlantic frequency and
intensity changes against AMOC change, and stacked mean frequency / SAT blocks.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .core import select_gwl, make_norm, plot_map_panel, add_colorbar
from .extremes import EXT_TYPE_PARAMS, threshold_symb

__all__ = ["plot_freq_and_intens_amoc_stereo", "plot_mean_freq_and_tas_stacked"]


def _draw_amoc_grid(subfig, items, amoc_changes, cmap, levels, cbar_label, panel_label, title):
    """2x4 North Polar Stereo grid (titled with each model's AMOC change) plus colorbar."""
    axes = subfig.subplots(2, 4, subplot_kw={"projection": ccrs.NorthPolarStereo()})
    cmap, norm = make_norm(levels, cmap)
    for (model, da), ax in zip(items, axes.flat):
        mesh = ax.pcolormesh(da.lon, da.lat, da, transform=ccrs.PlateCarree(),
                             cmap=cmap, norm=norm, shading="auto")
        ax.set_title(f"{model}, $\\Delta$ = {amoc_changes[model]:.2f} Sv", fontsize=16)
        ax.set_extent([-100, 60, 30, 90], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=False)
    add_colorbar(subfig, mesh, rect=[0.92, 0.08, 0.012, 0.72], label=cbar_label,
                 orientation="vertical", extend="both")
    subfig.text(0.01, 0.98, panel_label, fontsize=20, va="top", ha="left")
    subfig.suptitle(title, fontsize=25, y=0.99)
    subfig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.02, wspace=0.05, hspace=0.05)

def _select_extrema(da, key):
    """Pick `key` from the extrema dimension, if it hasn't been selected already."""
    return da.sel(extrema=key) if "extrema" in da.dims else da

def plot_freq_and_intens_amoc_stereo(ext_type, freq_differences, intens_differences,
                                     gwl, threshold, freq_levels, intens_levels,
                                     amoc_changes, period=None):
    """
    (a) exceedance-frequency change and (b) extreme-month SAT change, North
    Atlantic view. `ext_type` is 'Heat' or 'Cold'; `period` is 'final' or 'gwl'.
    """
    key = ext_type.lower()
    when = "by End of Simulation" if period == "final" else f"at {gwl}°C GWL"

    fig = plt.figure(figsize=(16, 13))
    top, bottom = fig.subfigures(2, 1, hspace=0.02)

    freq_items = [(m, select_gwl(freq_differences[m][threshold], gwl, period))
                  for m in freq_differences]
    _draw_amoc_grid(top, freq_items, amoc_changes, EXT_TYPE_PARAMS[key]["cmap2"], freq_levels,
                    "Difference in % of years", "(a)",
                    f"Change in {threshold_symb(key, threshold)} {ext_type} Extreme "
                    f"Exceedance Frequency {when}")

    intens_items = [(m, select_gwl(_select_extrema(ds.tas, key), gwl, period))
                    for m, ds in intens_differences.items()]
    _draw_amoc_grid(bottom, intens_items, amoc_changes, "RdBu_r", intens_levels,
                    "SAT Anomaly ($^o$C)", "(b)",
                    f"Change in {ext_type.capitalize()}est Month Temperature {when}")
    return fig


def _draw_2x2_robinson(subfig, rows, panel_label):
    """
    2x2 Robinson grid. `rows` = two dicts with keys
    data ({title: DataArray}), cmap, vmin, vmax, cbar_label.
    """
    axes = subfig.subplots(2, 2, subplot_kw={"projection": ccrs.Robinson()})
    cbar_boxes = [[0.93, 0.52, 0.012, 0.38], [0.93, 0.06, 0.012, 0.38]]
    for row, ax_row, box in zip(rows, axes, cbar_boxes):
        for (name, da), ax in zip(row["data"].items(), ax_row):
            im = plot_map_panel(ax, da, row["cmap"], vmin=row["vmin"], vmax=row["vmax"],
                                title=name, title_fontsize=18, method="imshow")
        add_colorbar(subfig, im, rect=box, label=row["cbar_label"],
                     orientation="vertical", extend="both")
    subfig.text(0.005, 0.99, panel_label, fontsize=20, va="top", ha="left")
    subfig.subplots_adjust(left=0.02, right=0.91, top=0.92, bottom=0.03, wspace=0.05, hspace=0.18)


def plot_mean_freq_and_tas_stacked(freq_rows, tas_rows):
    """(a) frequency 2x2 block on top, (b) SAT 2x2 block below."""
    fig = plt.figure(figsize=(16, 18))
    top, bottom = fig.subfigures(2, 1, hspace=0.02)
    _draw_2x2_robinson(top, freq_rows, "(a)")
    _draw_2x2_robinson(bottom, tas_rows, "(b)")
    return fig