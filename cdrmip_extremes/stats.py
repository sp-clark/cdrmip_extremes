import xarray as xr
import numpy as np
import cftime
from collections import defaultdict

def regress_pattern(X,Y,time_dim):
    """
    Perform regression of tas against gmst for one member.
    Returns the slope pattern (lat, lon).
    
    """

    # Ordinary least squares slope = cov(X, Y) / var(X)
    cov = xr.cov(X, Y, dim=time_dim)
    var = X.var(dim=time_dim)
    slope = cov / var
    
    return slope

def regress_pattern_with_r2(X,Y, time_dim="time"):
    """
    Perform regression of tas against gmst for xarray objects.
    Returns slope (lat, lon) and r^2 (lat, lon).
    """

    # Regression slope = cov(X,Y) / var(X)
    cov = xr.cov(X, Y, dim=time_dim)
    var = X.var(dim=time_dim)
    slope = cov / var

    # Predicted tas from regression
    Y_hat = slope * X

    # Residuals
    residuals = Y - Y_hat

    # Sum of squares
    ss_res = (residuals**2).sum(dim=time_dim)
    ss_tot = ((Y - Y.mean(dim=time_dim))**2).sum(dim=time_dim)

    r2 = 1 - ss_res / ss_tot

    return slope, r2