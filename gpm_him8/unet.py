import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
#from keras.utils import Sequence
import numpy as np
from tensorflow.keras import backend as K
import xarray as xr
from datetime import datetime
from datetime import timedelta
import pickle
import os
import math
from nc_loader import HIM8_GPM_Dataset


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


def get_unet():
    concat_axis = 3
    inputs = layers.Input(shape = (512, 512, 2))

    feats = 4#16
    bn0 = BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2) #256

    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4) #128

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6) #64

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8) #32

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = BatchNormalization(axis=3)(conv5)
    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = BatchNormalization(axis=3)(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10) #16

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12) #32
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)

    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv6)
    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn12)
    bn14 = BatchNormalization(axis=3)(conv6)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn10) #64
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)

    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn16) #128
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = BatchNormalization(axis=3)(conv7)

    # Rectify last convolution layer to constraint output to positive precipitation values.
    conv8 = layers.Conv2D(1, (1, 1), activation='relu')(bn14)

    model = models.Model(inputs=inputs, outputs=conv8)


    return model

train_gen = HIM8_GPM_Dataset("201811", batch_size=32)
test_gen = HIM8_GPM_Dataset("201812", batch_size=32)

model = get_unet()
print(model.summary())
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=conv_loss, metrics=['mse',mean_y,mean_yhat,var_y,var_yhat], optimizer=sgd)

print(tf.config.experimental.list_physical_devices('GPU'))
history = model.fit(train_gen, epochs=100, validation_data=test_gen)

with open('train_history_convtweak.pkl', 'wb') as f:
    pickle.dump(history.history, f)

model.save('gpm_model_5convwb100epochs.h5')

"""
for w_mean in [.5, 1, 2, 4]:
    for w_std in [.5, 1, 2, 4]:

        train_gen = HIM8_GPM_Dataset(batch_size=32)

        model = get_unet()
        print(model.summary())
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)

        history = model.fit_generator(generator=train_gen, epochs=50)

        with open('train_history_{}_{}.pkl'.format(w_mean, w_std), 'wb') as f:
            pickle.dump(history.history, f)

        model.save('gpm_model_{}_{}.h5'.format(w_mean, w_std))
"""
