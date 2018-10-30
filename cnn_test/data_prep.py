import netCDF4
import numpy as np
import os
import scipy.ndimage
from datetime import timedelta, datetime

start_date = datetime(2017, 1, 1)
end_date = datetime(2017, 2, 1)

himb1_stack = None
himb2_stack = None
himb3_stack = None
himb4_stack = None
himb5_stack = None
himb6_stack = None
himb7_stack = None
himb8_stack = None
himb9_stack = None
himb10_stack = None
himb11_stack = None
himb12_stack = None
himb13_stack = None
himb14_stack = None
himb15_stack = None
himb16_stack = None
gpm_stack = None

while start_date <= end_date:
    f_gpm = "/g/data/fj4/scratch/him8_cnn/GPM_{0}_SW_AU.nc".format(start_date.strftime("%Y%m%d%H%M"))
    f_him = "/g/data/fj4/scratch/him8_cnn/HIM8_{0}_SW_AU.nc".format(start_date.strftime("%Y%m%d%H%M"))
    if not os.path.isfile(f_him) or not os.path.isfile(f_gpm):
        print("timestamp", start_date, "not found")
        start_date = start_date + timedelta(minutes=30)
        continue

    with netCDF4.Dataset(f_gpm, 'r', format='NETCDF4') as src:
        # Output: 504, 952
        # Output: 504, 1000
        upsmpld = scipy.ndimage.zoom(src["precipitationCal"][:], [5.04, 5], order=1)
        print(upsmpld.shape, "has to be 504,1000")
        if gpm_stack is None:
            gpm_stack = upsmpld
        else:
            gpm_stack = np.vstack((gpm_stack, upsmpld))

    with netCDF4.Dataset(f_him, 'r', format='NETCDF4') as src:
        print(f_him)
        print(src)
        if himb1_stack is None:
            himb1_stack = np.expand_dims(src["B1"][:], axis=0)
            himb2_stack = np.expand_dims(src["B2"][:], axis=0)
            himb3_stack = np.expand_dims(src["B3"][:], axis=0)
            himb4_stack = np.expand_dims(src["B4"][:], axis=0)
            himb5_stack = np.expand_dims(src["B5"][:], axis=0)
            himb6_stack = np.expand_dims(src["B6"][:], axis=0)
            himb7_stack = np.expand_dims(src["B7"][:], axis=0)
            himb8_stack = np.expand_dims(src["B8"][:], axis=0)
            himb9_stack = np.expand_dims(src["B9"][:], axis=0)
            himb10_stack = np.expand_dims(src["B10"][:], axis=0)
            himb11_stack = np.expand_dims(src["B11"][:], axis=0)
            himb12_stack = np.expand_dims(src["B12"][:], axis=0)
            himb13_stack = np.expand_dims(src["B13"][:], axis=0)
            himb14_stack = np.expand_dims(src["B14"][:], axis=0)
            himb15_stack = np.expand_dims(src["B15"][:], axis=0)
            himb16_stack = np.expand_dims(src["B16"][:], axis=0)
        else:
            himb1_stack = np.vstack((himb1_stack, np.expand_dims(src["B1"][:], axis=0)))
            himb2_stack = np.vstack((himb2_stack, np.expand_dims(src["B2"][:], axis=0)))
            himb3_stack = np.vstack((himb3_stack, np.expand_dims(src["B3"][:], axis=0)))
            himb4_stack = np.vstack((himb4_stack, np.expand_dims(src["B4"][:], axis=0)))
            himb5_stack = np.vstack((himb5_stack, np.expand_dims(src["B5"][:], axis=0)))
            himb6_stack = np.vstack((himb6_stack, np.expand_dims(src["B6"][:], axis=0)))
            himb7_stack = np.vstack((himb7_stack, np.expand_dims(src["B7"][:], axis=0)))
            himb8_stack = np.vstack((himb8_stack, np.expand_dims(src["B8"][:], axis=0)))
            himb9_stack = np.vstack((himb9_stack, np.expand_dims(src["B9"][:], axis=0)))
            himb10_stack = np.vstack((himb10_stack, np.expand_dims(src["B10"][:], axis=0)))
            himb11_stack = np.vstack((himb11_stack, np.expand_dims(src["B11"][:], axis=0)))
            himb12_stack = np.vstack((himb12_stack, np.expand_dims(src["B12"][:], axis=0)))
            himb13_stack = np.vstack((himb13_stack, np.expand_dims(src["B13"][:], axis=0)))
            himb14_stack = np.vstack((himb14_stack, np.expand_dims(src["B14"][:], axis=0)))
            himb15_stack = np.vstack((himb15_stack, np.expand_dims(src["B15"][:], axis=0)))
            himb16_stack = np.vstack((himb16_stack, np.expand_dims(src["B16"][:], axis=0)))

    start_date = start_date + timedelta(minutes=30)

np.savez("/g/data/fj4/scratch/x", himb1_stack, himb2_stack, himb3_stack, himb4_stack, 
                                  himb5_stack, himb6_stack, himb7_stack, himb8_stack, 
                                  himb9_stack, himb10_stack, himb11_stack, himb12_stack, 
                                  himb13_stack, himb14_stack, himb15_stack, himb16_stack)

np.savez("/g/data/fj4/scratch/y", gpm_stack.data)
