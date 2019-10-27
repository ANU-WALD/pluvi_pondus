import numpy as np
import xarray as xr

temp = 15 + 8 * np.random.randn(2, 2, 24)
precip = 10 * np.random.rand(2, 2, 24)
lon = np.array([-99.83, -99.32])
lat = np.array([42.25, 42.21])
time = np.arange('2000-01-01', '2000-01-02', np.timedelta64(1, 'h'), dtype='datetime64')

ds = xr.Dataset({'temperature': (['lon', 'lat', 'time'],  temp),
                 'precipitation': (['lon', 'lat', 'time'], precip)},
                 coords={'lon': ('lon', lon),
                         'lat': ('lat', lat),
                         'time': ('time', time)})

ds.time.encoding['units'] = 'seconds since 1970-01-01'
ds.time.encoding['calendar'] = 'gregorian'

ds.to_netcdf("foo_ds.nc")
