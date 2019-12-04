import numpy as np
import xarray as xr

import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from datetime import timedelta



def model2nc(drange):
    print(drange)
    stack = None
    tstack = None

    #mse = load_model('../small_z/gan_mse{}_rainfields_generator.h5'.format(drange))
    mse = load_model('../small_z/unet_mse_rainfields.h5')

    for i in range(16,30):
        print(i)
    
        d = datetime(2018, 11, i+1, 0, 0)

        h8_fp = "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_{}.nc".format(d.strftime("%Y%m%d"))
        h8_ds = xr.open_dataset(h8_fp)

        b8 = h8_ds.B8.data[:,::2, ::2]
        b14 = h8_ds.B14.data[:,::2, ::2]
    

        x = np.stack((b8,b14), axis=-1)
        y = np.zeros((b8.shape[:3]), dtype=np.float32)

        for i in range(int(y.shape[0]/4)+1):
            if 4*i == y.shape[0]:
                break
            
            if 4*(i+1) >= y.shape[0]:
                y[4*i:,:-1,:-201] = mse.predict(x[4*i:,:-1,:-201,:])[:,:,:,0]
                y[4*i:,1:,201:] = mse.predict(x[4*i:,1:,201:,:])[:,:,:,0]
            
            else:
                y[4*i:4*(i+1),:-1,:-201] = mse.predict(x[4*i:4*(i+1),:-1,:-201,:])[:,:,:,0]
                y[4*i:4*(i+1),1:,201:] = mse.predict(x[4*i:4*(i+1),1:,201:,:])[:,:,:,0]
 
        if stack is None:
            stack = y
            tstack = h8_ds.time.values
        else:
            stack = np.concatenate((stack,y), axis=0)
            tstack = np.concatenate((tstack, h8_ds.time.values), axis=0)
        
        h8_ds.close()
    

    h8_fp = "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181101.nc"
    ds = xr.open_dataset(h8_fp)

    ds = ds.drop(["albers_conical_equal_area","B8","B14"])
    ds['time'] = tstack
    ds['y'] = ds['y'].values[::2]
    ds['x'] = ds['x'].values[::2]
    ds['prec'] = (('time', 'y', 'x'), stack)

    ds.to_netcdf("out_mse_{}_11h22018.nc".format(drange))



#dranges = ['1924', '2530']
dranges = ['0106']

for drange in dranges:
    model2nc(drange)
