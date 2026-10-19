import glob
import os 
import sys
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROC_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'
FIG_DIR = BASE_DIR / 'figures'

CDR_MODELS= ["ACCESS-ESM1-5","CanESM5","CESM2", "CNRM-ESM2-1","GFDL-ESM4","MIROC-ES2L","NorESM2-LM","UKESM1-0-LL"]

SSP_MODELS = [
    'ACCESS-CM2',
    'ACCESS-ESM1-5',
    'CESM2-WACCM',
    'CNRM-ESM2-1',
    'CanESM5',
    'EC-Earth3-Veg',
    'FGOALS-g3',
    'GFDL-ESM4',
    'GISS-E2-1-G',
    'GISS-E2-1-H',
    'GISS-E2-2-G',
    'IPSL-CM6A-LR',
    'MIROC-ES2L',
    'MPI-ESM1-2-LR',
    'MRI-ESM2-0',
    'NorESM2-LM',
    'UKESM1-0-LL'
]


CDR_EXPTS = ['1pctCO2','1pctCO2-cdr','piControl']

# define colour scheme
cdr_colours = ['forestgreen','orange','blue','red','purple','springgreen','pink','dodgerblue']
cdr_colour_dict = {model:cdr_colours[i] for i, model in enumerate(CDR_MODELS)}
cdr_colour_dict['Multi-Model Median'] = 'black'

# ssp_colours = plt.cm.tab10.colors + ('gold', 'black', 'teal', 'magenta')
# ssp_colour_dict = {model: ssp_colours[i] for i, model in enumerate(SSP_MODELS)}

EXTRA_COLOURS = ['brown', 'navy', 'teal', 'cyan', 'olive', 'darkkhaki', 'slategrey',
                 'magenta', 'mediumorchid','gold','salmon', 'grey',]

def make_colour_dict(models):
    colours = dict(cdr_colour_dict)
    new = [m for m in models if m not in colours]
    if len(new) > len(EXTRA_COLOURS):
        raise ValueError(f'{len(new)} new models but only {len(EXTRA_COLOURS)} extra colours')
    colours.update(zip(new, EXTRA_COLOURS))
    return colours

COLOUR_DICT = make_colour_dict(SSP_MODELS)

# loading arguments
DEFAULT_CHUNKS = {'time': 365}
DEFAULT_VARS_TO_DROP = ['height', 'time_bnds', 'lat_bnds', 'lon_bnds']


CDR_members = {
    'ACCESS-ESM1-5': 'r1i1p1f1',
    'CESM2': 'r1i1p1f1',
    'CNRM-ESM2-1': 'r1i1p1f2',
    'CanESM5': 'r1i1p2f1',
    'GFDL-ESM4': 'r1i1p1f1',
    'MIROC-ES2L': 'r1i1p1f2',
    'NorESM2-LM': 'r1i1p1f1',
    'UKESM1-0-LL': 'r1i1p1f2'
}