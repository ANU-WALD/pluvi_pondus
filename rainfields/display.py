import xarray as xr
import numpy as np
import imageio
from datetime import datetime
from datetime import timedelta

dates = [datetime(2018,11,1,12,0),datetime(2018,11,2,15,0),datetime(2018,11,3,18,0),datetime(2018,11,4,12,0)]

for d in dates:
    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(d.strftime("%Y%m%d")))
    h8 = ds["B8"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data
    imageio.imwrite('H8_{}.png'.format(d.strftime("%Y%m%d")), h8)

    dp = datetime(2018,11,d.day,0,0)
    rf = None
    for i in range(24*6):
        #print(i)
        ds = xr.open_dataset("/data/pluvi_pondus/Rainfields/{}/310_{}_{}00.prcp-c10.nc".format(int(d.strftime("%d")), d.strftime("%Y%m%d"), dp.strftime("%H%M")))
        if rf is None:
            rf = np.nan_to_num(ds["precipitation"].data)
            print(rf.dtype, rf.min(), rf.max())
        else:
            rf += np.nan_to_num(ds["precipitation"].data)
            print("  ", rf.dtype, rf.min(), rf.max())
            a = ds["precipitation"].data
            print("  ", float(100*np.count_nonzero(np.isnan(a)))/a.size)
            print("  ", np.count_nonzero(~np.isnan(a)))

        dp = dp + timedelta(0,60*10)

    imageio.imwrite('RF_{}.png'.format(d.strftime("%Y%m%d")), rf)
