import numpy as np
import xarray as xr

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from datetime import datetime
from datetime import timedelta


def conv_loss(y_true, y_pred):
    n = 5
    mk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32) / n**2, dtype='float32')
    sk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32), dtype='float32')
    
    Bias = K.mean(K.abs(K.conv2d(y_true, mk)-K.conv2d(y_pred, mk)), axis=-1)
    Var = K.mean(K.abs((K.conv2d(K.square(y_true), sk) - K.square(K.conv2d(y_true, sk)/n**2))/n**2 - (K.conv2d(K.square(y_pred), sk) - K.square(K.conv2d(y_pred, sk)/n**2))/n**2), axis=-1)
    
    return Var + 4*Bias


def mean_y(y_true, y_pred):
    n = 5
    mk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32) / n**2, dtype='float32')
    
    Mean = K.mean(K.conv2d(y_true, mk), axis=-1)

    return Mean


def mean_yhat(y_true, y_pred):
    n = 5
    mk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32) / n**2, dtype='float32')
    
    Mean = K.mean(K.conv2d(y_pred, mk), axis=-1)

    return Mean


def var_y(y_true, y_pred):
    n = 5
    sk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32), dtype='float32')
    
    Var = K.mean((K.conv2d(K.square(y_true), sk) - K.square(K.conv2d(y_true, sk)/n**2))/n**2, axis=-1)


    return Var


def var_yhat(y_true, y_pred):
    n = 5
    sk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32), dtype='float32')
    
    Var = K.mean((K.conv2d(K.square(y_pred), sk) - K.square(K.conv2d(y_pred, sk)/n**2))/n**2, axis=-1)

    return Var


stack = None
tstack = None
model = load_model('gpm_model_mse100epochs.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})
#model = load_model('gpm_model_5conv100epochs.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})

ds = xr.open_dataset("/data/GPM_HIM8/HIM8_201812.nc")
print(ds)


for i, t in enumerate(ds.time):

    b8 = (ds['B8'].sel(time=t).values)#/(256-196)
    b14 = (ds['B14'].sel(time=t).values)#/(327-196)
    
    x = np.stack((b8,b14), axis=-1)
    pred = model.predict(x[None,:,:,:])[:,:,:,0]

    if stack is None:
        stack = pred
    else:
        stack = np.concatenate((stack,pred), axis=0)
    print(stack.shape)

            
print(stack.shape, stack.max())
  
ds = ds.drop(["albers_conical_equal_area","B8","B9","B10","B11","B12","B13","B14","B15","B16"])
ds['y'] = ds['y'].values[::4]
ds['x'] = ds['x'].values[::4]
print(ds)
ds['prec'] = (('time', 'y', 'x'), stack)

ds.to_netcdf("mse_out.nc")
