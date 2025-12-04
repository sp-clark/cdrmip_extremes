# CDRMIP Extremes #
This repository comprises the necessary code to reproduce the analysis of Clark et al. (submitted) examining changes in monthly temperature extremes within the Carbon Dioxide Removal Model Intercomparison Project (CDRMIP) Tier 1 experiment. The repository has the following structure:

├── cdrmip_extremes
│   ├── configs.py
│   ├── ext_freq.py
│   ├── __init__.py
│   ├── load_data.py
│   ├── load_esgf.py
│   ├── plotting
│   │   ├── plot_extremes.py
│   │   └── plot_sat.py
│   ├── sat.py
│   └── utils.py
├── data
│   ├── processed
│   └── raw
│       ├── mrsos
│       ├── msftmz-msftyz
│       ├── siconc
│       └── tas
├── environment.yml
├── notebooks
│   ├── 01_calculate_GSAT.ipynb
│   ├── 02_GSAT_metrics.ipynb
│   ├── 03_SAT_calculations.ipynb
│   ├── 04_Figure1.ipynb
│   ├── 05_define_thresholds.ipynb
│   ├── 06_calc_ext_frequencies.ipynb
│   ├── 07_Figures_2_to_5.ipynb
│   ├── 08_FigureS1.ipynb
│   ├── 09_Figures_S3_to_S6.ipynb
│   ├── 10_load_amoc.ipynb
│   ├── 10_model_disagreement.ipynb
│   ├── 11_calc_amoc.ipynb
│   ├── 12_plot_amoc.ipynb
│   ├── 13_storylines.ipynb
│   ├── 14_storylines_final_period.ipynb
│   ├── 15_soil_moisture.ipynb
│   └── 16_sea_ice.ipynb
└── README.md

The modules necessary to perform the analysis are contained within the subdirectory cdrmip_extremes, while the code to reproduce the analysis and figures of the submitted paper are included as jupyter notebooks that are designed to be completed sequentially. 

Note that in order to complete the analysis, data must first be downloaded from the Earth System Grid Federation (https://aims2.llnl.gov/search). The required netcdf files are described in Clark et al. (submitted). Surface Air Temperature ('tas') data, moreover, should be regridded for each model onto a common 2ºC by 2ºC latitude-longitude grid. This can be done, for instance, through cdo operators (https://code.mpimet.mpg.de/projects/cdo). To perform the analysis, this data should be stored within the 'data/raw' directory, which should in turn contain subdirectories for each required variable ('tas','msftmz-msftyz','mrsos' and 'siconc'), within which are subdirectories for the three required experiments ('1pctCO2','1pctCO2-cdr', and 'piControl'), which in turn contain the necessary files from each Earth System Model included in the analysis.

The required packages to perform the analysis are furthermore specified in the environment.yml file. To install this run the following commands:

conda env create -f environment.yml
conda activate cdrmip_extremes


