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


def generate_output(h5_model, date_str):
    conv_model = load_model(h5_model, custom_objects={'conv_loss': conv_loss,'mean_y':mean_y,'mean_yhat':mean_yhat,'var_y':var_y,'var_yhat':var_yhat})

    dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_{}.nc".format(date_str))

    stack = None

    for i, t in enumerate(dsh.time):

        b8 = dsh['B8'].sel(time=t).values
        b8 = (b8-237)/6.5
        b14 = dsh['B14'].sel(time=t).values
        b14 = (b14-280)/20

        x = np.stack((b8,b14), axis=-1)
        y_hat = conv_model.predict(x[None,:,:,:])[:,:,:,0]
    
        if stack is None:
            stack = y_hat
        else:
            stack = np.concatenate((stack,y_hat), axis=0)

    dsg = xr.open_dataset("/data/GPM_HIM8/GPM_{}.nc".format(date_str))
    dsg = dsg.drop(["PrecCal"])
    dsg["time"] = dsh.time
    dsg["PrecCal"] = (('time','y','x'), stack)
    dsg.to_netcdf(h5_model[:-3]+"_{}.nc".format(date_str))


generate_output('model_2months_150epochs_mse.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha1_beta1.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha2_beta1.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha4_beta1.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha1_beta2.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha2_beta2.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha4_beta2.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha1_beta4.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha2_beta4.h5', '201903')
generate_output('model_2months_150epochs_conv5_alpha4_beta4.h5', '201903')
