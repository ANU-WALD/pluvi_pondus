
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
from nc_loader_small import HimfieldsDataset

def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        print(l.name)
        print(l.output_shape)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def mse_custom(y_true, y_pred):
    N = 9
    mk = K.constant(value=np.ones((3,3,1,1), dtype=np.float32) / N, dtype='float32')
    MSconvE = K.mean(K.square(K.conv2d(y_true, mk)-K.conv2d(y_pred, mk)), axis=-1)

    #sk = K.constant(value=np.ones((3,3,1,1), dtype=np.float32), dtype='float32')
    #MSconvSD = K.mean(K.square((vK.conv2d(K.square(y_true), sk) - (K.square(K.conv2d(y_true, sk))/N))/N - (vK.conv2d(K.square(y_pred), sk) - (K.square(K.conv2d(y_pred, sk))/N))/N), axis=-1)


    return MSconvE #+ MSconvSD

def get_unet():
    concat_axis = 3
    inputs = layers.Input(shape = (512, 512, 2))

    feats = 16
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

    conv8 = layers.Conv2D(1, (1, 1))(bn14)

    model = models.Model(inputs=inputs, outputs=conv8)


    return model


train_fnames = ["/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181101.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181102.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181103.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181104.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181105.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181106.nc"]

training_gen = HIM8_GPM_Dataset(train_fnames, 1, batch_size=8)
validation_gen = HIM8_GPM_Dataset(test_fnames, 1, batch_size=8)

model = get_unet()
print(model.summary())
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=mse_custom, optimizer=sgd)

history = model.fit_generator(generator=training_gen, validation_data=validation_gen, epochs=100, verbose=1)

with open('train_history_him8_8batch.pkl', 'wb') as f:
    pickle.dump(history.history, f)

parallel_model.save('rainfields_model_mse_small.h5')
