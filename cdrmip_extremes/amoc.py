"""
amoc.py
"""
import numpy as np

LAT_DIMS = ('lat', 'rlat', 'y', 'j-mean')
LAT_OVERRIDES = {'CNRM-ESM2-1': 190.283}   # j-mean index nearest 26.5N

def _standardise(ds):
    """Return the streamfunction DataArray with dims (time, basin, lev, <lat>)."""
    da = ds['msftmz'] if 'msftmz' in ds else ds['msftyz']
    # IPSL: (time, 3basin, olevel, y, x=1) with 2D nav_lat
    da = da.rename({k: v for k, v in {'olevel': 'lev', '3basin': 'basin'}.items() if k in da.dims})
    if 'x' in da.dims:
        da = da.squeeze('x', drop=True).assign_coords(y=ds['nav_lat'].squeeze('x').values)
    return da

def _atlantic(da, ds):
    """Select the Atlantic basin by its label where one exists, else index 0."""
    for name in ('sector', 'region'):
        if name in ds:
            labels = [s.decode() if isinstance(s, bytes) else str(s) for s in ds[name].values]
            return da.isel(basin=next(i for i, s in enumerate(labels) if 'atlantic' in s.lower()))
    return da.isel(basin=0)

def calc_amoc(ds, lat=26.5):
    da = _atlantic(_standardise(ds), ds)
    lat_dim = next(d for d in da.dims if d in LAT_DIMS)
    target = LAT_OVERRIDES.get(ds.attrs['source_id'], lat)
    idx = int(abs(da[lat_dim] - target).argmin())
    return (da.isel({lat_dim: idx}).max('lev') / 1.025e9).rename('amoc')   # kg/s -> Sv