import numpy as np
import xarray as xr
import os
import netCDF4 as nc
import cftime
from pathlib import Path

from cdrmip_extremes.configs import data_dir, models, expts


def load_raw_tas():
    tas_dir = os.path.join(data_dir,'raw/tas')
    data = {model:{} for model in models}
    for model in models:
        for expt in expts:
            path = os.path.join(
                tas_dir,
                expt,
                f"{model}_{expt}_Amon_tas_r180x90.nc"
            )
            data[model][expt] = xr.open_dataset(path,drop_variables=['height'])
    return data