import xarray as xr
import numpy as np
import imageio
from datetime import datetime
from datetime import timedelta

d = datetime(2018,11,1,0,10)
x_0 = []
y_0 = []
x = []
y = []

#while d < datetime(2018,12,1,0,0):
while d < datetime(2018,11,4,0,0):
    ds = xr.open_dataset("/data/pluvi_pondus/Rainfields/{}/310_{}_{}00.prcp-c10.nc".format(int(d.strftime("%d")), d.strftime("%Y%m%d"), d.strftime("%H%M")))
    rf = ds["precipitation"].data#.flatten()

    ds.close()

    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(d.strftime("%Y%m%d")))
    h8 = ds["B8"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data
    h14 = ds["B14"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data
    ds.close()

    dp = d - timedelta(0,10)
    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(dp.strftime("%Y%m%d")))
    h8p = ds["B8"].sel(time=datetime(dp.year,dp.month,dp.day,dp.hour,0)).data
    h14p = ds["B14"].sel(time=datetime(dp.year,dp.month,dp.day,dp.hour,0)).data
    ds.close()

    prec_vals_0 = []
    him8_tiles_0 = []
    prec_vals = []
    him8_tiles = []
    for j in range(2, rf.shape[0]):
        for i in range(2, rf.shape[1]):
            if ~np.isnan(rf[j,i]):
                if rf[j,i] == 0 and len(prec_vals_0) < 2*(len(prec_vals_0)+1):
                    prec_vals_0.append(rf[j,i])
                    him8_tiles_0.append(np.dstack((h8p[j-2:j+3,i-2:i+3], h14p[j-2:j+3,i-2:i+3], h8[j-2:j+3,i-2:i+3], h14[j-2:j+3,i-2:i+3])))
                else:
                    prec_vals.append(rf[j,i])
                    him8_tiles.append(np.dstack((h8p[j-2:j+3,i-2:i+3], h14p[j-2:j+3,i-2:i+3], h8[j-2:j+3,i-2:i+3], h14[j-2:j+3,i-2:i+3])))

    x.append(np.stack(him8_tiles, axis=0))
    x.append(np.stack(him8_tiles_0, axis=0))
    y.append(np.array(prec_vals))
    y.append(np.array(prec_vals_0))

    print(d.strftime("%Y%m%dT%H%M"))
    d += timedelta(0,60*10)

print(x[0].shape)
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)
print("  ", x.shape, y.shape)

np.save("x_conv", x)
np.save("y_conv", y)
