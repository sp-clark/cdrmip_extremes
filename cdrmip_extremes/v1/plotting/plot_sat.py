import os
import sys
import xarray as xr
import pandas as pd
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# define colour scheme
colours = ['forestgreen','orange','blue','red','purple','springgreen','pink','dodgerblue']
colour_dict = {model:colours[i] for i, model in enumerate(models)}


def plot_gmst_and_gwls(gmst_da,fig,ax):

    for model in models:
        da = gmst_da.sel(model=model)
        da_rolling = da.rolling(year=21,center=True,min_periods=1).mean()
        da.plot(ax=ax,color=colour_dict[model],linewidth=2,alpha=0.3)
        da_rolling.plot(ax=ax,label=model,color=colour_dict[model],linewidth=2)

        # plot 1.5 crossing year
        gwl_up = GWL_crossing_years.sel(model=model).sel(simulation='ramp_up').sel(gwl=1.5)
        gwl_down = GWL_crossing_years.sel(model=model).sel(simulation='ramp_down').sel(gwl=1.5)

        ax.scatter(
            gwl_up,
            gwl_up.gwl,
           marker='^',
          color=colour_dict[model],
          s=400,
          # label='1.5$^o$C ramp-up'
            zorder=10
        )
        ax.scatter(gwl_down,
           gwl_down.gwl,
           marker='v',
          color=colour_dict[model],
          s=400,
          # label='1.5$^o$C ramp-down'
        zorder=10
                  )
    
    ax.set_ylabel('GSAT Anomaly (°C)',fontsize=18)
    ax.set_xlim([0,345])
    x_ticks = [0,140,280,340]  # Add 340 to the list of x-ticks
    x_tick_labels = ['0','140','280','340']  # Corresponding tick labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels,fontsize=16)
    ax.set_xlabel('Year',fontsize=18)

    ax.axvline(x = 140, color = 'gray',linewidth=1.0,linestyle='--',alpha=0.7)
    ax.axvline(x = 280, color = 'gray',linewidth=1.0,linestyle='--',alpha=0.7)
    ax.axhline(y = 0, color = 'black',linewidth=1.0)
    
    ax2 = ax.twinx()
    co2_line = ax2.plot(CO2[100:],
                        label='Atmospheric CO$_2$',
                         color='gray',
                         linewidth=2,
                         linestyle='--',
                         alpha=0.7
                       )
    ax2.set_ylabel('Atmospheric CO$_2$ (ppm)',fontsize=18,color='gray')
    ax2.tick_params(axis='y', colors='gray')                # Gray ticks and tick marks
    ax2.yaxis.label.set_color('gray')  

    ax.tick_params(axis='y', labelsize=16)    # For the main y-axis (GSAT)
    ax2.tick_params(axis='y', labelsize=16) 

    ax.set_title('GSAT',fontsize=21)

    handles, labels = ax.get_legend_handles_labels()
    # Append the CO₂ line to the legend

    handles2,labels2 = ax2.get_legend_handles_labels()
    # labels2 = []
    # handles2.append(co2_line[0])
    # labels2.append('Atmospheric CO$_2$')
    

    # Create a legend below all subplots
    ax.legend(handles, labels, loc='upper left', 
              # bbox_to_anchor=(0.5, -0.1), 
              ncol=1, fontsize=12,frameon=False)
    ax2.legend(handles2,labels2,loc='upper right',ncol=1,fontsize=12,frameon=False)