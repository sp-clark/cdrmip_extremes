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

def load_tas_concat():
    
    save_dir = os.path.join(data_dir,'processed/tas/concatenated')
    data = {}
    for model, ds in concat.items():
        path = os.path.join(
            save_dir,
            f"{model}_cdr-reversibility_tas_concat.nc"
        )
        data[model] = xr.open_dataset(path)

def load_tas_anom():
    data = {}
    save_dir = os.path.join(data_dir,'processed/tas/anomalies')
    for model in models:
        path = os.path.join(
            save_dir,
            f"{model}_cdr-reversibility_tas_anom.nc"
        )
        data[model] = xr.open_dataset(path)
    return data

def load_gsat():
    gsat = {}
    # save gsat
    save_dir = os.path.join(data_dir,'processed/gsat')
    for model in models:
        path = os.path.join(
            save_dir,
            f"{model}_cdr-reversibility_gsat.nc"
        )
        gsat[model] = xr.open_dataarray(path)
    return gsat

def load_gwl_years():
    save_dir = os.path.join(
        data_dir,"processed/gwl_years"
    )
    gwl_years = {}
    for model in models:
        path = os.path.join(
            save_dir,
            f"{model}_gwl_years.nc"
        )
        gwl_years[model] = xr.open_dataarray(path)
    return gwl_years

def load_equiv_gwls():
    path = os.path.join(data_dir,"processed/revised_gwl/matched_gwls.nc")
    return xr.open_dataset(path)

def load_threshold_data():
    threshold_data = {model:{} for model in models}
    save_dir = os.path.join(
        data_dir,'processed/extremes'
    )
    ext_vars = ['heat_thresholds', 'cold_thresholds']
    for model in models:
        for var in ext_vars:
            var_dir = os.path.join(save_dir,var)
            path = os.path.join(
                var_dir,
                f"{model}_{var}.nc"
            )
            threshold_data[model][var] = xr.open_dataset(path)
    return threshold_data

def load_monthly_extreme_data():
    data = {model:{} for model in models}
    save_dir = os.path.join(
        data_dir,'processed/extremes'
    )
    ext_vars = ['max_month_mean','max_month_std_dev','min_month_mean','min_month_std_dev']
    for model in models:
        for var in ext_vars:
            var_dir = os.path.join(save_dir,var)
            path = os.path.join(
                var_dir,
                f"{model}_{var}.nc"
            )
            data[model][var] = xr.open_dataarray(path)
    return data
    