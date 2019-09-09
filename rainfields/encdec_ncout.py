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
        d += timedelta(0,10*60)
        continue
            
    b8 = h8_ds.B8.sel(time=d).data
    b14 = h8_ds.B14.sel(time=d).data
    b8p = h8p_ds.B8.sel(time=dp).data
    b14p =h8p_ds.B14.sel(time=dp).data

    h8_ds.close()
    h8p_ds.close()
    
    x = np.stack((b8p,b14p,b8,b14), axis=-1)

    arr[2:,402:] = model.predict(x[None,2:,402:,:])
    arr[:-2,:-402] = model.predict(x[None,:-2,:-402,:])

    return arr


model = load_model('rainfields_model.h5', custom_objects={'mse_holes': mse_holes})

