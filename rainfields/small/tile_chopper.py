import xarray as xr
import numpy as np

ds = xr.open_dataset("gan.nc")
ds.sel(x=slice(1231000,2253000),y=slice(-3247000,-4269000)).to_netcdf("syd_tile.nc")
