import xarray as xr
import numpy as np
import imageio
from datetime import datetime
from datetime import timedelta

d = datetime(2018,11,1,12,10)
x = []
y = []

#while d < datetime(2018,12,1,0,0):
while d < datetime(2018,11,7,0,0):
    ds = xr.open_dataset("/data/pluvi_pondus/Rainfields/{}/310_{}_{}00.prcp-c10.nc".format(int(d.strftime("%d")), d.strftime("%Y%m%d"), d.strftime("%H%M")))
    rf = ds["precipitation"].data#.flatten()
    print(rf.shape)
    exit()

    ds.close()
    mask = ~np.isnan(rf)
    y.append(rf[mask].flatten())

    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(d.strftime("%Y%m%d")))
    h8 = ds["B8"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data
    h14 = ds["B14"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data
    ds.close()

    dp = d - timedelta(0,10)
    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(dp.strftime("%Y%m%d")))
    h8p = ds["B8"].sel(time=datetime(dp.year,dp.month,dp.day,dp.hour,0)).data
    h14p = ds["B14"].sel(time=datetime(dp.year,dp.month,dp.day,dp.hour,0)).data
    ds.close()

    chunk = np.stack((h8[mask].flatten(), h8p[mask].flatten(), h14[mask].flatten(), h14p[mask].flatten()), axis=1)
    print(chunk.shape)
    x.append(chunk)

    print(d.strftime("%Y%m%dT%H%M"))
    d += timedelta(0,60*10)

x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)
print("  ", x.shape, y.shape)

np.save("xp", x)
np.save("yp", y)
