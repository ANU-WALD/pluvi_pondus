import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
import numpy as np
from tensorflow.keras import backend as K
from datetime import datetime
from datetime import timedelta
import pickle
import os
import math

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

y_train = np.clip(y_train, 0, 40)
y_test = np.clip(y_test, 0, 40)

print("MSE train", np.mean(np.square(y_train)))
print("MSE test", np.mean(np.square(y_test)))


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



def get_unet():
    concat_axis = 3
    inputs = layers.Input(shape = (512, 512, 2))

    feats = 4#16
    bn0 = BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2) #256

    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4) #128

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6) #64

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn8 = BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8) #32

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn10 = BatchNormalization(axis=3)(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10) #16

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn11) #32
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)

    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn13) #64
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)

    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn15) #128
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)

    # Rectify last convolution layer to constraint output to positive precipitation values.
    conv8 = layers.Conv2D(1, (1, 1), activation='relu')(bn13)

    model = models.Model(inputs=inputs, outputs=conv8)


    return model


losses = {'msle': mean_squared_log_error, 'mse': mean_squared_error, }
losses = {'ms4e': mean_fourth_error}

for name, loss in losses.items():
    model = get_unet()
    print(model.summary())
    opt = Adagrad(lr=0.0001)
    model.compile(loss=loss, metrics=[mean_squared_error, mean_squared_log_error, 'mae'], optimizer=opt)

    print(tf.config.experimental.list_physical_devices('GPU'))
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

    with open('history_3months_100epochs_4chan_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('model_3months_100epochs_4chan_{}.h5'.format(name))
