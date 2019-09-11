import numpy as np 
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from datetime import datetime
from datetime import timedelta
import os
import xarray as xr

def mse_holes(y_true, y_pred):
    idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    y_true = K.tf.gather_nd(y_true, idxs)
    y_pred = K.tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

def get_himfields(model, d):
    arr = np.zeros((2050, 2450), dtype=np.float32)

    dp = d - timedelta(0, 10*60)
    h8_fp = "/data/pluvi_pondus/2B/HIM8_2B_AU_{}.nc".format(d.strftime("%Y%m%d"))
    h8p_fp = "/data/pluvi_pondus/2B/HIM8_2B_AU_{}.nc".format(dp.strftime("%Y%m%d"))

    if not os.path.exists(h8_fp) or not os.path.exists(h8p_fp):
        return arr

    h8_ds = xr.open_dataset(h8_fp)
    h8p_ds = xr.open_dataset(h8p_fp)

    if np.datetime64(d) not in h8_ds.time.data or np.datetime64(dp) not in h8p_ds.time.data:
        return arr
            
    b8 = h8_ds.B8.sel(time=d).data
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
model = load_model('rainfields_model.h5', custom_objects={'mse_holes': mse_holes})

start = datetime(2018, 11, 1)
while start <= datetime(2018, 12, 31):
    print (start.strftime("%Y-%m-%d"))

    ds = xr.open_dataset("/data/pluvi_pondus/2B/HIM8_2B_AU_{}.nc".format(start.strftime("%Y%m%d")))
    ds0 = ds.copy(deep=True)
    ds.close()

    arr = np.zeros((ds0.time.data.shape[0], 2050, 2450), dtype=np.float32)
    ds0 = ds0.drop(["B8","B14"])

    for i, d in enumerate(ds0.time.data):
        dt = datetime.utcfromtimestamp(d.astype(int) * ns)
        print(i, d, dt)
        arr[i,:,:] = get_himfields(model, dt)
        print("--", arr.max())
    
    ds0['himfields'] = (('time', 'y', 'x'), arr)
    ds0.to_netcdf("/data/pluvi_pondus/Himfields_{}.nc".format(start.strftime("%Y%m%d")))
    ds0.close()
    start += timedelta(days=1)
