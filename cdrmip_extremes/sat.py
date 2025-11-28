import xarray as xr
import numpy as np
import cftime


def find_crossing_years(
    gsat_da,
    window,
    gwls,
    time_dim='year',
    overshoot=True,
) -> list:
    
    # define rolling timeseries
    if time_dim == 'year':
        rolling = gsat_da.rolling(year=window,center=True,min_periods=1).mean()
    else:
        rolling = (
            gsat_da
                .groupby('time.year').mean(dim='time')
                .rolling(year=window,center=True,min_periods=1).mean()
        )
    crossing_years = {}
    if overshoot:
        # separate either side of peak warming
        peak_year = int(rolling.idxmax(dim='year'))
        pre_peak = rolling.sel(year=slice(None,peak_year))
        post_peak = rolling.sel(year=slice(peak_year+1,None))

        for gwl in gwls:
            # make mask around specified gwl for each branch and use it to identify crossing years
            pre_peak_cross = int(pre_peak.where(pre_peak>=gwl).dropna(dim='year').year[0])
            post_peak_cross = int(post_peak.where(post_peak<=gwl).dropna(dim='year').year[0])
            
            crossing_years[gwl] = {'ramp_up':pre_peak_cross,'ramp_down':post_peak_cross}
    else:
        for gwl in gwls:
            # assume we have a warming timeseries
            cross = int(rolling.where(rolling>=gwl).drop_na(dim='year').year[0])
            crossing_years[gwl] = [cross]

    da = xr.DataArray(
        [[year for year in crossing_years[gwl].values()] for gwl in crossing_years.keys()],
        coords={
            'gwl':list(crossing_years.keys()),
            'branch':['ramp_up','ramp_down']
        },
        dims=['gwl','branch']
    )

    return da

def find_matching_gwls(
    gsat_da,
    window,
    match_slice,
    time_dim='year'
):
    # define rolling timeseries
    if time_dim == 'year':
        rolling = gsat_da.rolling(year=window,center=True,min_periods=1).mean()
    else:
        rolling = (
            gsat_da
                .groupby('time.year').mean(dim='time')
                .rolling(year=window,center=True,min_periods=1).mean()
        )
    # identify end gwl
    end_gwl = np.round(
        float(rolling.sel(year=match_slice).mean(dim='year').values),
        2
    )

    # isolate ramp_up
    ramp_up = rolling.isel(year=slice(0,140))

    # identify crossing year
    exceed_year = int(ramp_up.where(ramp_up>=end_gwl).dropna(dim='year')['year'][0])

    return [end_gwl, exceed_year]

