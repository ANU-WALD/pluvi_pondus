import numpy as np
import xarray as xr
from datetime import datetime
from datetime import timedelta
  
ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181101.nc")
x = ds.x.data[::2]
y = ds.y.data[::2]
ds.close()

ds = xr.open_dataset("/data/pluvi_pondus/Rainfields/310_20181101_000000.prcp-c10.nc")
proj = ds.proj
ds.close()

model = "gan"
for d in range(1,31):
  date = datetime(2018, 11, d)
  precip = np.load("{}_{}.npy".format(model, d))
  time = np.arange(date.strftime('%Y-%m-%d'), (date+timedelta(days=1)).strftime('%Y-%m-%d'), np.timedelta64(10, 'm'), dtype='datetime64')

  ds = xr.Dataset({'precipitation': (['time', 'y', 'x'], precip)},
                   coords={'time': ('time', time),
                           'y': ('y', y),
                           'x': ('x', x)})

  ds.time.encoding['units'] = 'seconds since 1970-01-01'
  ds.time.encoding['calendar'] = 'gregorian'
  ds['proj'] = proj
  
  ds.resample(time='0.5H').reduce(np.mean).to_netcdf("{}_{:02d}.nc".format(model, d))
  
  prec = np.nan_to_num(ds.precipitation.values)
  prec = np.where(prec > .1, np.log(prec) - np.log(.1), 0)
  ds['precipitation'] = (['time', 'y', 'x'], prec)

  ds.to_netcdf("{}_{:02d}.nc".format(model, d))

"""
model = "mse"
for d in range(1,31):
  date = datetime(2018, 11, d)
  precip = np.load("{}_{}.npy".format(model, d))
  time = np.arange(date.strftime('%Y-%m-%d'), (date+timedelta(days=1)).strftime('%Y-%m-%d'), np.timedelta64(10, 'm'), dtype='datetime64')

  ds = xr.Dataset({'precipitation': (['time', 'y', 'x'], precip)},
                   coords={'time': ('time', time),
                           'y': ('y', y),
                           'x': ('x', x)})

  ds.time.encoding['units'] = 'seconds since 1970-01-01'
  ds.time.encoding['calendar'] = 'gregorian'

  ds.to_netcdf("{}_{:02d}.nc".format(model, d))
"""
