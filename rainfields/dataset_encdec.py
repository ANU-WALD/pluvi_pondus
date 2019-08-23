import xarray as xr
import numpy as np
import imageio
from datetime import datetime
from datetime import timedelta

d = datetime(2018,11,1,0,10)
x = []
y = []

#while d < datetime(2018,12,1,0,0):
while d < datetime(2018,11,7,0,0):
    ds = xr.open_dataset("/data/pluvi_pondus/Rainfields/{}/310_{}_{}00.prcp-c10.nc".format(int(d.strftime("%d")), d.strftime("%Y%m%d"), d.strftime("%H%M")))
    rf = ds["precipitation"].data[1:-1, 400:-2][::2,::2]
    ds.close()

    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(d.strftime("%Y%m%d")))
    h8 = ds["B8"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data[1:-1, 400:-2][::2,::2]
    h14 = ds["B14"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data[1:-1, 400:-2][::2,::2]
    ds.close()

    dp = d - timedelta(0,10)
    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(dp.strftime("%Y%m%d")))
    h8p = ds["B8"].sel(time=datetime(dp.year,dp.month,dp.day,dp.hour,0)).data[1:-1, 400:-2][::2,::2]
    h14p = ds["B14"].sel(time=datetime(dp.year,dp.month,dp.day,dp.hour,0)).data[1:-1, 400:-2][::2,::2]
    ds.close()

    x.append(np.stack((h8p,h14p,h8,h14), axis=-1))
    print(x[0].shape)
    y.append(rf[:,:,None])
    print(y[0].shape)

    print(d.strftime("%Y%m%dT%H%M"))
    d += timedelta(0,6*60*10)

x = np.stack(x, axis=0)
y = np.stack(y, axis=0)
print("  ", x.shape, y.shape)

np.save("x_ed", x)
np.save("y_ed", y)
