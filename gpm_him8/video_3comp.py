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


def mean_squared_error(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def mean_fourth_error(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.pow(y_pred - y_true, 4), axis=-1)

def mean_squared_log_error(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(K.log(1+y_pred) - K.log(1+y_true)), axis=-1)

def mean_squared_exp_error(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(K.exp(y_pred) - K.exp(y_true)), axis=-1)


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

x = np.load("x_test.npy")
xz = np.load("xz_test.npy")
y = np.clip(np.load("y_test.npy"),0,30)[:,:,:,0]
print(y.shape, y.min(), y.max(), y.mean(), y.var())

model = load_model('model_3months_100epochs_4chan_mse.h5', custom_objects={'mean_squared_error': mean_squared_error, 'mean_squared_log_error': mean_squared_log_error})
y_hat_mse = model.predict([x])[:,:,:,0]

model = load_model('model_3months_100epochs_4chan_ms4e.h5', custom_objects={'mean_squared_error': mean_squared_error, 'mean_squared_log_error': mean_squared_log_error, 'mean_fourth_error': mean_fourth_error})
y_hat_conv = model.predict(x)[:,:,:,0]


writer = imageio.get_writer('him_gpm.mp4', fps=4)

for i in range(y.shape[0]):
    stack = np.hstack((y[i,:,:],y_hat_mse[i,:,:],y_hat_conv[i,:,:]))
    stack = ((np.clip(stack, 0, 30)/30)*255).astype(np.uint8)
    stack = 255 - stack
    stack[:,128] = 0
    stack[:,256] = 0

    writer.append_data(stack)

writer.close()
