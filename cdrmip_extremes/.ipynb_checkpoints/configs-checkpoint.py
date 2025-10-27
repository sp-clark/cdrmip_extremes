import glob
import os 
import sys
from pathlib import Path

data_dir = Path('/g/data/if69/sc1326/CDRMIP/data')

models = ["ACCESS-ESM1-5","CanESM5","CESM2", "CNRM-ESM2-1","GFDL-ESM4","MIROC-ES2L","NorESM2-LM","UKESM1-0-LL"]


expts = ['1pctCO2','1pctCO2-cdr','piControl']