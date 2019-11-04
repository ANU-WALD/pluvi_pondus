import numpy as np 
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from datetime import datetime
from datetime import timedelta
import os
import xarray as xr


def get_himfields(model, d):
    arr = np.zeros((2050, 2450), dtype=np.float32)

    dp = d - timedelta(0, 10*60)
    h8_fp = "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(d.strftime("%Y%m%d"))
    h8p_fp = "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(dp.strftime("%Y%m%d"))

    if not os.path.exists(h8_fp) or not os.path.exists(h8p_fp):
        return arr

    h8_ds = xr.open_dataset(h8_fp)
    h8p_ds = xr.open_dataset(h8p_fp)

    if np.datetime64(d) not in h8_ds.time.data or np.datetime64(dp) not in h8p_ds.time.data:
        return arr
            
    b8 = h8_ds.B8.sel(time=d).data
    print(b8)
    b14 = h8_ds.B14.sel(time=d).data
    b8p = h8p_ds.B8.sel(time=dp).data
    b14p =h8p_ds.B14.sel(time=dp).data

    h8_ds.close()
    h8p_ds.close()
    
    x = np.stack((b8p,b14p,b8,b14), axis=-1)
    print(x.shape, x.min(), x.max(), x[0,0,0]) 
    
    arr[:-2,:-402] = model.predict(x[None,:-2,:-402,:])[0,:,:,0]
    arr[2:,402:] = model.predict(x[None,2:,402:,:])[0,:,:,0]
    print(arr.max())
    
    return np.clip(arr, 0, None)


ns = 1e-9 # number of seconds in a nanosecond
model = load_model('rainfields_model_mae.h5', custom_objects={'mae_holes': mae_holes})

start = datetime(2018, 11, 2)
while start <= datetime(2018, 11, 2):
    print (start.strftime("%Y-%m-%d"))

    ds = xr.open_dataset("/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(start.strftime("%Y%m%d")))
    ds0 = ds.copy(deep=True)
    ds.close()

    arr = np.zeros((ds0.time.data.shape[0], 2050, 2450), dtype=np.float32)
    ds0 = ds0.drop(["B8","B14"])

    for i, d in enumerate(ds0.time.data):
        dt = datetime.utcfromtimestamp(d.astype(int) * ns)
        print(i, d, dt)
        arr[i,:,:] = get_himfields(model, dt)
        print("--", arr.max())
   
    np.save("arr.npy", arr)
    exit() 
    ds0['himfields'] = (('time', 'y', 'x'), arr)
    ds0.to_netcdf("Himfields_mae_{}.nc".format(start.strftime("%Y%m%d")))
    ds0.close()
    start += timedelta(days=1)
