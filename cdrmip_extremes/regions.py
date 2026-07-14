import os
import sys
import xarray as xr
import pandas as pd
import netCDF4 as nc
import numpy as np
import regionmask

ar6_all = regionmask.defined_regions.ar6.all
ar6_land = regionmask.defined_regions.ar6.land
ar6_ocean = regionmask.defined_regions.ar6.ocean

region_dict = dict(zip(ar6_all.abbrevs,ar6_all.numbers))
region_names = dict(zip(ar6_all.abbrevs,ar6_all.names))


def regionmask_average(ds,land_mask,ocean_mask,surface):
    if surface == 'all':
        ar6_surface = ar6_all
    if surface == 'land':
        ar6_surface = ar6_land
    if surface == 'ocean':
        ar6_surface = ar6_ocean
    mask_3D = ar6_surface.mask_3D(ds)
    weights = np.cos(np.deg2rad(ds.lat))
    
    regional = (ds).weighted(mask_3D*weights).mean(dim=('lat','lon'))
    land_regional = (ds.where(land_mask==1)).weighted(mask_3D*weights).mean(dim=('lat','lon'))
    ocean_regional = (ds.where(ocean_mask==1)).weighted(mask_3D*weights).mean(dim=('lat','lon'))

    return regional, land_regional, ocean_regional

def land_ocean_masks(sftlf):
    land_mask = xr.where(sftlf > 50, 1, 0)   # 1 = land, 0 = ocean
    ocean_mask = xr.where(sftlf <= 50, 1, 0) # 1 = ocean, 0 = land
    return land_mask,ocean_mask

def zonal_average(ds, lower_bound, upper_bound, mask=None):
    """Compute weighted zonal average between latitude bounds, optionally applying a mask."""
    subset = ds.where((ds.lat >= lower_bound) & (ds.lat < upper_bound))
    
    if mask is not None:
        subset = subset.where(mask==1)
        
    weights = np.cos(np.deg2rad(ds.lat))
    
    return subset.weighted(weights).mean(dim=('lat', 'lon'))

zonal_lat_bounds = [(-90,0),(0,90),(-90.0,-60.0),(-60.0,-30.0),(-30.0,30.0),(30.0,60.0),(60.0,90.0)]
zonal_regions = ["Southern Hemisphere","Northern Hemisphere","Antarctic","SH Extratropics","Tropics","NH Extratropics","Arctic"]
zonal_dict = dict(zip(zonal_regions,zonal_lat_bounds))

def get_zonal_averages(ds,mask=None):
    zonal_avgs = {}
    for zone, bounds in zonal_dict.items():
        zonal_avg = zonal_average(ds,bounds[0],bounds[1],mask)
        zonal_avgs[zone] = zonal_avg
    return zonal_avgs

def zonal_subset(ds, lower_bound, upper_bound, mask=None):
    """Return ds masked to lats in [lower_bound, upper_bound), optionally applying a mask."""
    subset = ds.where((ds.lat >= lower_bound) & (ds.lat < upper_bound))
    if mask is not None:
        subset = subset.where(mask == 1)
    return subset

def get_zonal_subsets(ds, mask=None):
    return {zone: zonal_subset(ds, *bounds, mask) for zone, bounds in zonal_dict.items()}

def area_fraction_above(subset, threshold=0):
    """Percent of (weighted) region area where each variable exceeds threshold.

    Works on the NaN-masked output of zonal_subset; NaN cells are
    excluded from both numerator and denominator.
    """
    weights = np.cos(np.deg2rad(subset.lat))
    indicator = (subset > threshold).where(subset.notnull())  # 1/0 inside region, NaN outside
    return indicator.weighted(weights).mean(dim=('lat', 'lon')) * 100

def area_fraction_below(subset, threshold=0):
    """Percent of (weighted) region area where each variable exceeds threshold.

    Works on the NaN-masked output of zonal_subset; NaN cells are
    excluded from both numerator and denominator.
    """
    weights = np.cos(np.deg2rad(subset.lat))
    indicator = (subset < threshold).where(subset.notnull())  # 1/0 inside region, NaN outside
    return indicator.weighted(weights).mean(dim=('lat', 'lon')) * 100