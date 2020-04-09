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

print("MSE train", np.mean(np.square(y_train)))
print("MSE test", np.mean(np.square(y_test)))

def conv_loss_gen(alpha, beta, n):
    def conv_loss(y_true, y_pred):
        mk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32) / n**2, dtype='float32')
        sk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32), dtype='float32')
    
        Bias = K.mean(K.square(K.conv2d(y_true, mk) - K.conv2d(y_pred, mk)), axis=-1)
        Var = K.mean(K.abs((K.conv2d(K.square(y_true), sk) - K.square(K.conv2d(y_true, sk))/n**2)/n**2 - (K.conv2d(K.square(y_pred), sk) - K.square(K.conv2d(y_pred, sk))/n**2)/n**2), axis=-1)
    
        return alpha*Var + beta*Bias

    return conv_loss


def mean_y(y_true, y_pred):
    n = 9
    mk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32) / n**2, dtype='float32')
    
    Mean = K.mean(K.conv2d(y_true, mk), axis=-1)

    return Mean


def mean_yhat(y_true, y_pred):
    n = 9
    mk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32) / n**2, dtype='float32')
    
    Mean = K.mean(K.conv2d(y_pred, mk), axis=-1)

    return Mean


def var_y(y_true, y_pred):
    n = 9
    sk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32), dtype='float32')
    
    Var = K.mean((K.conv2d(K.square(y_true), sk) - K.square(K.conv2d(y_true, sk)/n**2))/n**2, axis=-1)

    return Var

def var_yhat(y_true, y_pred):
    n = 9
    sk = K.constant(value=np.ones((n,n,1,1), dtype=np.float32), dtype='float32')
    
    Var = K.mean((K.conv2d(K.square(y_pred), sk) - K.square(K.conv2d(y_pred, sk)/n**2))/n**2, axis=-1)

    return Var


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


losses = {'mse':'mse'}

"""
losses = {}

#for conv_size in [7,9]:
for conv_size in [9]:
    for alpha in [4.]:
        for beta in [1.]:
            losses["conv{}_alpha{}_beta{}".format(conv_size,int(alpha),int(beta))] = conv_loss_gen(alpha,beta,conv_size)
"""


for name, loss in losses.items():
    model = get_unet()
    print(model.summary())
    opt = Adagrad(lr=0.0001)#, decay=1e-6)
    model.compile(loss=loss, metrics=['mse',mean_y,mean_yhat,var_y,var_yhat], optimizer=opt)

    print(tf.config.experimental.list_physical_devices('GPU'))
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200)

    with open('history_3months_200epochs_4chan_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('model_3months_200epochs_4chan_{}.h5'.format(name))
