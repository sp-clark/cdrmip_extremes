import os
import sys
import xarray as xr
import pandas as pd
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm


def plot_pattern(expt,period,patterns,var,ext_type=None):
    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 6),
        subplot_kw={'projection': ccrs.Robinson()}
    )
    
    if var == 'slope':
        levels = np.linspace(0,2,17)
        cmap = cm.RdBu_r
    elif var == 'r2':
        levels = np.linspace(-1,1,17)
        cmap = cm.viridis
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    
    
    for (model,ds_dict), ax in zip(patterns.items(),axes.flat):
        if ext_type is not None:
            ds = ds_dict[var].sel(extrema=ext_type)
        else:
            ds = ds_dict[var]
        im = ax.imshow(
            ds,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            origin='lower',
            extent=(0, 360, -90, 90),
        )
        ax.coastlines()
        ax.set_title(model, fontsize=16)
    
    cbar_ax = fig.add_axes([0.1, -0.05, 0.8, 0.04])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
        extend="both"
    )
    cbar.set_label('Local GSAT dependence (°C/°C)', fontsize=14)
    # cbar.set_ticks([-1,-0.5,0,0.5,1])
    cbar.ax.tick_params(labelsize=12)
    
    fig.suptitle(f"{expt} {period} warming regressions {var}",fontsize=24)
    
    fig.tight_layout()