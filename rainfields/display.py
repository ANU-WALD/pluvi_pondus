import xarray as xr
import imageio
from datetime import datetime
dates = [datetime(2018,11,1,12,0),datetime(2018,11,2,15,0),datetime(2018,11,3,18,0),datetime(2018,11,4,12,0)]

for d in dates:
    ds = xr.open_dataset("/data/pluvi_pondus/HIM8_AU_{}.nc".format(d.strftime("%Y%m%d")))
    h8 = ds["B8"].sel(time=datetime(d.year,d.month,d.day,d.hour,0)).data
    imageio.imwrite('H8_{}.png'.format(d.strftime("%Y%m%d")), h8)

    ds = xr.open_dataset("/data/pluvi_pondus/Rainfields/{}/310_{}_{}00.prcp-c10.nc".format(int(d.strftime("%d")), d.strftime("%Y%m%d"), d.strftime("%H%M")))
    rf = ds["precipitation"].data
    imageio.imwrite('RF_{}.png'.format(d.strftime("%Y%m%d")), rf)
