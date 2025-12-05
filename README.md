# cdrmip_extremes #
This repository comprises the necessary code to reproduce the analysis of Clark et al. (submitted) examining changes in monthly temperature extremes within the Carbon Dioxide Removal Model Intercomparison Project (CDRMIP) Tier 1 experiment. 

The repository has the following structure:

```
├── cdrmip_extremes
├── data
│   ├── processed
│   └── raw
│       ├── mrsos
│       ├── msftmz-msftyz
│       ├── siconc
│       └── tas
├── environment.yml
├── notebooks
└── README.md
```
### How to use this code ###
1. Clone this repository to your local system
2. Create the necessary conda environment using the included environment.yml file through the following commands:
    ```
    conda env create -f environment.yml
    conda activate cdrmip_extremes
    ```
3. To perform the analysis, the required data must first be pre-downloaded from the Earth System Grid Federation (ESGF, https://aims2.llnl.gov/search) and stored within the ```data/raw``` subdirectory. The necessary ESGF files are described in Clark et al. (submitted). Note that Surface Air Temperature ('tas') data should be regridded for each model onto a common 2ºC by 2ºC latitude-longitude grid prior to performing the analysis. This can be done, for instance, through cdo operators (https://code.mpimet.mpg.de/projects/cdo).
4. Run through the notebooks within the ```notebooks``` subdirectory sequentially. Each notebook will load in the necessary modules from ```cdrmip_extremes``` and the required data from the ```data``` subdirectory.




