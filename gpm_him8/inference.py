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

conv_model1 = load_model('model_2months_200epochs_4chan_conv3_alpha1_beta2.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})
conv_model2 = load_model('model_2months_200epochs_4chan_conv5_alpha1_beta2.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})
conv_model4 = load_model('model_2months_200epochs_4chan_conv7_alpha1_beta2.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})
mse_model = load_model('model_2months_200epochs_4chan_mse.h5', custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})

dsg = xr.open_dataset("/data/GPM_HIM8/GPM_201903.nc")
dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_201903.nc")

print(dsg.time)

#Select rainy period in December
dsg = dsg.sel(time=slice(datetime(2019,3,21),datetime(2019,3,31)))

writer = imageio.get_writer('him_gpm.mp4', fps=4)

for i, t in enumerate(dsg.time):
    d = datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
    if np.datetime64(d) not in dsh.time.values:
        continue

    prec = dsg['PrecCal'].sel(time=t).values
    b8 = dsh['B8'].sel(time=t).values
    b8 = (b8-237)/6.5
    b14 = dsh['B14'].sel(time=t).values
    b14 = (b14-280)/20

    x = np.stack((b8,b14), axis=-1)
    y = prec[:, :, None]
    #plt.imsave("GPM_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(y[:,:,0], 0, 50), vmin=0, vmax=50, cmap=cm)

    y_hat_conv1 = conv_model1.predict(x[None,:,:,:])[:,:,:,0]
    y_hat_conv2 = conv_model2.predict(x[None,:,:,:])[:,:,:,0]
    y_hat_conv4 = conv_model4.predict(x[None,:,:,:])[:,:,:,0]
    #print(t, y.shape, y.max(), y_hat.shape, y_hat.max())
    #plt.imsave("Conv5_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(y_hat_conv1[0,:,:], 0, 50), vmin=0, vmax=50, cmap=cm)

    y_hat_mse = mse_model.predict(x[None,:,:,:])[:,:,:,0]
    #print(t, y.shape, y.max(), y_hat.shape, y_hat.max())
    #plt.imsave("MSE_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(y_hat_mse[0,:,:], 0, 50), vmin=0, vmax=50, cmap=cm)

    stacku = np.hstack((y[:,:,0],y_hat_mse[0,:,:],y_hat_conv1[0,:,:]))
    stackd = np.hstack((y[:,:,0],y_hat_conv2[0,:,:],y_hat_conv4[0,:,:]))
    stack = np.vstack((stacku,stackd))
    stack = ((np.clip(stack, 0, 20)/20)*255).astype(np.uint8)
    stack = 255 - stack
    stack[128,:] = 0
    stack[:,128] = 0
    stack[:,256] = 0

    writer.append_data(stack)

writer.close()
