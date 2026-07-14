import xarray as xr
import numpy as np
import cftime
from collections import defaultdict

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
    
def rolling(ds,window,time_dim='year'):
    if time_dim != 'year':
        ds = ds.groupby('time.year').mean(dim='time')
    return ds.rolling(year=window,center=True,min_periods=1).mean()


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def _map_leaves(nested_dict, func):
    out = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            out[k] = _map_leaves(v, func)
        else:
            out[k] = func(v)
    return out

def sort_models(nested_dict, model_order):
    known = [m for m in model_order if m in nested_dict]
    unknown = [m for m in nested_dict if m not in model_order]
    if unknown:
        print(f"  sort_models: no order defined for {unknown}, appending at end")
    return {m: nested_dict[m] for m in known + unknown}
    
def build_co2_path(piCo2=284.7):
    CO2_4x = piCo2 * 4
    years_up = np.arange(0,140,1)
    concs_up = piCo2 * np.power(1.01, years_up)
    concs_up = np.append(concs_up, CO2_4x)
    years_up = np.append(years_up, 139.321)
    years_down = np.array(139.321)
    years_down = np.append(years_down,np.arange(140,280, 1))
    concs_down = concs_up[0:-1][::-1]
    concs = np.concatenate((concs_up,concs_down),axis=None)
    years = np.concatenate((years_up,years_down),axis=None)
    PI = piCo2 * np.ones(100)
    yearsPI = np.arange(0,100,1)
    const = piCo2 * np.ones(60)
    years_const = np.arange(281,341,1)
    CO2_series = np.concatenate(
        [PI,
         concs[:140],
         concs[140:],
         const],
        axis=None
    )
    years = np.concatenate(
        [yearsPI,
         years[:140]+100,
         years[141:]+100,
         years_const+100,
        ],
        axis=None
    )
    return xr.DataArray(data=CO2_series,
             dims=['year'],
             coords={'year':years}
            )

def flatten_single(data):
    out = {}
    for model, expt_dict in data.items():
        (expt, member_dict), = expt_dict.items()
        (member, val), = member_dict.items()
        out[model] = val
    return out


