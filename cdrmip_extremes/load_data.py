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
            data[model][expt] = xr.open_dataset(path,drop_variables=['height','time_bnds'])
    return data

def load_tas_concat():
    
    save_dir = os.path.join(data_dir,'processed/tas/concatenated')
    data = {}
    for model in models:
        path = os.path.join(
            save_dir,
            f"{model}_cdr-reversibility_tas_concat.nc"
        )
        data[model] = xr.open_dataset(path)
    return data

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

def load_sat_difference(period):
    save_dir = os.path.join(data_dir,'processed/tas',period)
    data = {}
    for model in models:
        path = os.path.join(
            save_dir,
            f"{model}_{period}_difference.nc"
        )
        data[model] = xr.open_dataset(path)
    return data

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
    path = os.path.join(data_dir,"processed/gwl_years/matched_gwls.nc")
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

def load_monthly_extreme_data(ext_vars=None):
    data = {model:{} for model in models}
    save_dir = os.path.join(
        data_dir,'processed/extremes'
    )
    if ext_vars == None:
        ext_vars = [
            'extreme_months',
            'max_month_mean',
            'max_month_std_dev',
            'min_month_mean',
            'min_month_std_dev'
        ]
    for model in models:
        for var in ext_vars:
            var_dir = os.path.join(save_dir,var)
            path = os.path.join(
                var_dir,
                f"{model}_{var}.nc"
            )
            if var == 'extreme_months':
                data[model][var] = xr.open_dataset(path)
            else:
                data[model][var] = xr.open_dataarray(path)
    return data

def load_ext_freq_data(final=False):
    model_list = ['Multi-Model Median'] + models
    data = {model:{} for model in model_list}
    save_dir = os.path.join(
        data_dir,'processed/extremes'
    )

    for var in ['cold_exceedances','heat_exceedances']:
        if final:
            var_dir = os.path.join(save_dir,f'{var}_final')
            path = os.path.join(
                var_dir,
                f"median_{var}_final.nc"
            )
        else:
            var_dir = os.path.join(save_dir,var)
            path = os.path.join(
                var_dir,
                f"median_{var}.nc"
            )
        data['Multi-Model Median'][var] = xr.open_dataset(path)

    for model in models:
        for var in ['cold_exceedances','heat_exceedances']:
            if final:
                var_dir = os.path.join(save_dir,f'{var}_final')
                path = os.path.join(
                    var_dir,
                    f"{model}_{var}_final.nc"
                )
            else:
                var_dir = os.path.join(save_dir,var)
                path = os.path.join(
                    var_dir,
                    f"{model}_{var}.nc"
                )
            data[model][var] = xr.open_dataset(path)
        
    return data

def load_ext_month_tas(final=False):
    data = {}
    if final:
        save_dir = os.path.join(
            data_dir,'processed/extremes/extreme_month_tas_final'
        )
        period='final'
    else:
        save_dir = os.path.join(
            data_dir,'processed/extremes/extreme_month_tas'
        )
        period='gwls'
    for model in models:
        path = os.path.join(
            save_dir,
            f"{model}_extreme_month_tas_{period}.nc"
        )
        data[model] = xr.open_dataset(path)
    return data

def load_amoc():
    amoc_dir = os.path.join(data_dir,'processed/amoc')
    amoc_data = {model:{} for model in models}
    for model in models:
        path = os.path.join(amoc_dir,f"{model}_amoc_26N.nc")
        path_pi = os.path.join(amoc_dir,f"{model}_amoc_26N_piControl.nc")
        amoc = xr.open_dataarray(path).groupby('time.year').mean(dim='time')
        amoc_piControl = xr.open_dataarray(path_pi).groupby('time.year').mean(dim='time')
        amoc_data[model]['amoc_26N'] = amoc
        amoc_data[model]['amoc_piControl'] = amoc_piControl
        amoc_data[model]['anom'] =  amoc - amoc_piControl.mean(dim='year')
        amoc_data[model]['std_dev'] = amoc_piControl.std(dim='year')
    return amoc_data
    