"""
temperature.py -- local peak-warming timing and SAT reversibility relative to GSAT.
"""

import math

import numpy as np
import matplotlib.cm as cm

from .core import (
    display_name, mmm_first, make_norm, make_geo_grid,
    plot_map_panel, add_colorbar, hide_unused_axes, maybe_save,
)

__all__ = ["plot_peak_warming_years", "plot_reversibility_rel_gsat_normalised"]

BOTTOM_CBAR = [0.1, -0.025, 0.8, 0.03]


def plot_peak_warming_years(peak_years, peak_gsat, expt, n_cols=3, levels=None,
                            cmap=None, figsize=None,
                            cbar_label="Local peak warming year rel. to GSAT peak warming year",
                            suptitle=None, fig_dir=None, folder=None, file_name=None):
    """Local peak-warming year minus the GSAT peak-warming year, one map per model."""
    models = mmm_first(m for m in peak_years if expt in peak_years[m])

    levels = np.arange(-40, 41, 10) if levels is None else levels
    cmap, norm = make_norm(levels, cmap or cm.RdBu_r)
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = make_geo_grid(n_rows, n_cols, figsize=figsize or (15, n_rows * 10 / 3))

    for model, ax in zip(models, axes.flat):
        peak_gsat_year = peak_gsat[model]["peak_year"]
        mesh = plot_map_panel(
            ax, peak_years[model][expt] - peak_gsat_year, cmap, norm=norm, method="imshow",
            title=f"{display_name(model)} | Peak GSAT year: {peak_gsat_year:.0f}",
        )
    hide_unused_axes(axes, len(models))

    add_colorbar(fig, mesh, rect=BOTTOM_CBAR, label=cbar_label, extend="both")
    fig.suptitle(suptitle or f"{expt} year of peak warming", fontsize=22, y=1.02)
    fig.tight_layout()
    maybe_save(fig, fig_dir, folder, file_name)
    return fig


def plot_reversibility_rel_gsat_normalised(reversibility, reversibility_gsat, window, bins,
                                           rev_year, expt, CO2, n_cols=3,
                                          fig_dir=None, folder=None, file_name=None):
    """Local SAT difference minus the GSAT difference at `rev_year`, one map per model."""
    models = mmm_first(reversibility)
    cmap, norm = make_norm(np.linspace(-window, window, bins), cm.RdBu_r)
    fig, axes = make_geo_grid(math.ceil(len(models) / n_cols), n_cols, figsize=(15, 10))

    for model, ax in zip(models, axes.flat):
        gsat_diff = np.round(reversibility_gsat[model][expt].sel(year=rev_year).values, 2)
        da = reversibility[model][expt].sel(year=rev_year) - gsat_diff
        mesh = plot_map_panel(ax, da, cmap, norm=norm, method="imshow",
                              title=f"{display_name(model)} | GSAT Difference: {gsat_diff}°C")
    hide_unused_axes(axes, len(models))

    add_colorbar(fig, mesh, rect=BOTTOM_CBAR, extend="both",
                 label="Local SAT difference rel. to GSAT difference (°C)")
    co2_value = int(np.round(CO2.sel(year=rev_year + 100).values))
    fig.suptitle(f"SAT Reversibility following return to {co2_value}ppm CO₂ (Year {280 - rev_year})",
                 fontsize=22, y=1.02)
    fig.tight_layout()
    maybe_save(fig, fig_dir, folder, file_name)
    return fig