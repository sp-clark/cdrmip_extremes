import xarray as xr
import numpy as np
import cftime



def global_mean(x, var="tas"):
    """
    Function to calculate cosine weighted global mean for a given variable
    """
    # Check whether input is a dataset or dataarray
    if isinstance(x, xr.Dataset):
        da = x[var]
    elif isinstance(x, xr.DataArray):
        da = x
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")

    # Weighing Data
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"

    return da.weighted(weights).mean(("lon", "lat"))

def global_surface_mean(x, var, surface):
    # Check whether input is a dataset or dataarray
    if isinstance(x, xr.Dataset):
        da = x[var]
    elif isinstance(x, xr.DataArray):
        da = x
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")

    if surface == 'land':
        weights=areacella
    elif surface == 'ocean':
        weights = areacello
    else:
        raise ValueError("surface must be 'land' or 'ocean'")

    return da.weighted(weights=weights).mean(dim=weights.dims)

def concat_branches(ds_up, ds_down):
    time_up = xr.cftime_range("0000-01-16",freq="1M",periods=12*140,calendar='noleap')
    len_down = len(ds_down.time)
    time_down = xr.cftime_range("0140-01-16",freq="1M",periods = len_down,calendar='noleap')

    ds_up = ds_up.isel(time=slice(None,12*140)).assign_coords({'time':time_up})
    ds_down = ds_down.assign_coords({'time':time_down})

    return xr.concat([ds_up,ds_down],dim='time')

def calc_anomaly(
    ds: xr.Dataset,
    ds_ref: xr.Dataset,
    reference_period: slice,
    *,
    time_dim="time",
) -> xr.Dataset:
    """
    Function to calculate anomaly with respect to 
    reference period. Assumes child dataset is already
    concatenated to parent.
    """
    # select reference slice and mean
    if time_dim not in ds.dims:
        raise ValueError(f"{time_dim} is not a dimension of this dataset. "
                         f"Available dims: {list(ds.dims)}")
    # calculate reference mean
    ref = ds_ref.sel({time_dim:reference_period})
    ref = ref.mean(dim=time_dim)

    # subtract reference from ds
    anom = ds - ref
    return anom


