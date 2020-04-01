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
xz_train = np.load("xz_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
xz_test = np.load("xz_test.npy")
y_test = np.load("y_test.npy")

print(y_train.max(), y_test.max())

y_train = np.clip(y_train,0,30)/30
y_test = np.clip(y_test,0,30)/30

print(y_train.max(), y_test.max())


print("MSE train", np.mean(np.square(y_train)))
print("MSE test", np.mean(np.square(y_test)))


def get_unet():
    concat_axis = 3
    ref_input = layers.Input(shape = (512, 512, 2))
    z_input = layers.Input(shape = (64, 64, 3))

    feats = 4#16
    bn0 = BatchNormalization(axis=3)(ref_input)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2) #256

    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4) #128

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6) #64
    
    zadd = layers.concatenate([z_input, pool3], axis=concat_axis)
    nzadd = layers.BatchNormalization(axis=3)(zadd)

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(nzadd)
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
    conv8 = layers.Conv2D(1, (1, 1), activation='sigmoid')(bn13)

    model = models.Model(inputs=[ref_input,z_input], outputs=conv8)


    return model

"""
#losses = {'mse':'mse'}


for conv_size in [3,5,7]:
    for alpha in [1.]:#,6.,8.]:
        #for beta in [4.,6.,8.]:
        #losses["conv{}_alpha{}_beta{}".format(conv_size,int(alpha),int(beta))] = conv_loss_gen(alpha,beta,conv_size)
        losses["conv{}_alpha{}_beta{}".format(conv_size,int(alpha),int(alpha))] = conv_loss_gen(alpha,alpha,conv_size)
"""

losses = {"kl": "kld"}

for name, loss in losses.items():
    model = get_unet()
    print(model.summary())
    opt = Adagrad(lr=0.0001)#, decay=1e-6)
    model.compile(loss=loss, metrics=['mse',mean_y,mean_yhat,var_y,var_yhat], optimizer=opt)

    print(tf.config.experimental.list_physical_devices('GPU'))
    history = model.fit((x_train, xz_train), y_train, validation_data=((x_test, xz_test), y_test), epochs=50)

    with open('history_3monthscm_100epochs_4chanz_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('model_3monthscm_100epochs_4chanz_{}.h5'.format(name))
