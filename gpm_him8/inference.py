import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import imageio
from datetime import datetime
from datetime import timedelta
import os
import xarray as xr
from itertools import chain


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


###################################### ColorMap ##############################################
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

color_list = np.array([(255, 255, 255),  # 0.0
                       (245, 245, 255),  # 0.2
                       (180, 180, 255),  # 0.5
                       (120, 120, 255),  # 1.5
                       (20,  20, 255),   # 2.5
                       (0, 216, 195),    # 4.0
                       (0, 150, 144),    # 6.0
                       (0, 102, 102),    # 10
                       (255, 255,   0),  # 15
                       (255, 200,   0),  # 20
                       (255, 150,   0),  # 30
                       (255, 100,   0),  # 40
                       (255,   0,   0),  # 50
                       (200,   0,   0),  # 60
                       (120,   0,   0),  # 75
                       (40,   0,   0)])  # > 100

color_list = color_list/255.
cm = LinearSegmentedColormap.from_list("BOM-RF3", color_list, N=32)

conv_model = load_model('gpm_model_5conv100epochs.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})
mse_model = load_model('gpm_model_mse100epochs.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})

dsg = xr.open_dataset("/data/GPM_HIM8/GPM_201812.nc")
dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_201812.nc")

#Select rainy period in December
dsg = dsg.sel(time=slice(datetime(2018,12,12),datetime(2018,12,17)))

for i, t in enumerate(dsg.time):
    d = datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
    if np.datetime64(d) not in dsh.time.values:
        continue

    prec = dsg['PrecCal'].sel(time=t).values
    b8 = (dsh['B8'].sel(time=t).values)#/(256-196)
    b14 = (dsh['B14'].sel(time=t).values)#/(327-196)

    x = np.stack((b8,b14), axis=-1)
    y = prec[:, :, None]
    plt.imsave("GPM_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(y[:,:,0], 0, 50), vmin=0, vmax=50, cmap=cm)

    y_hat = conv_model.predict(x[None,:,:,:])[:,:,:,0]
    print(t, y.shape, y.max(), y_hat.shape, y_hat.max())
    plt.imsave("Conv5_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(y_hat[0,:,:], 0, 50), vmin=0, vmax=50, cmap=cm)

    y_hat = mse_model.predict(x[None,:,:,:])[:,:,:,0]
    print(t, y.shape, y.max(), y_hat.shape, y_hat.max())
    plt.imsave("MSE_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(y_hat[0,:,:], 0, 50), vmin=0, vmax=50, cmap=cm)

