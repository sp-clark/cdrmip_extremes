import os
import sys
import xarray as xr
import pandas as pd
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


cbar_kwargs = {
    "fraction": 0.06,
    "aspect": 20,
    'orientation':'vertical',
    'location':'right',
    'anchor':(5,0.5),
    'pad':0.02,
}

def plot_hottest_coldest_month(
    extreme_months,
    extrema,
    title
):
    fig, axes = plt.subplots(
        2, 4,
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(16, 6),
        sharey=True
    )

    cmap = plt.cm.twilight_shifted 
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    boundaries = np.arange(1, 14)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    for (model, dt), ax in zip(extreme_months.items(),axes.flat):
        ds = dt.ds.sel(extrema=extrema).month
        im = ax.imshow(
            ds,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            origin='lower',
            extent=(0, 360, -90, 90)
        )
        ax.set_title(model, fontsize=15)
        ax.coastlines()

    # Add a single discrete colorbar for all subplots
    cbar = fig.colorbar(im, ax=axes, boundaries=boundaries, **cbar_kwargs)
    cbar.set_ticks(np.arange(1.5, 13.5))  # Center ticks within each bin
    cbar.set_ticklabels(month_labels)  # Set tick labels to month names
    cbar.set_label("Month", fontsize=12)
    fig.suptitle(title, fontsize=25, y=0.95)
    fig.tight_layout()
    plt.show()

def plot_pi_stats(
    ext_dict,
    variable,
    cmap,
    vmin,
    vmax,
    cbar_label,
    title
):
    fig, axes = plt.subplots(2,4, subplot_kw={"projection":ccrs.Robinson()},
                         figsize=(16,6),
                         sharey=True)

    cbar_kwargs['label'] = cbar_label
    if 'mean' in variable:
        cbar_kwargs['extend'] = 'both'
    else:
        cbar_kwargs['extend'] = 'max'

    for (model, ds_dict), ax in zip(ext_dict.items(),axes.flat):
        if 'mean' in variable:
            ds = ds_dict[variable] - 273.15
        else:
            ds = ds_dict[variable]
        im = ax.imshow(
            ds,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            extent=(0, 360, -90, 90)
        )
        ax.set_title(model,fontsize=15)
        ax.coastlines()
    cbar = fig.colorbar(im,ax=axes,**cbar_kwargs)

    fig.suptitle(title,fontsize=25,y=0.95)
    fig.tight_layout()
    plt.show()